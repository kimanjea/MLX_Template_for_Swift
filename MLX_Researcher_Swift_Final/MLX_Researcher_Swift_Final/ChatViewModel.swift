import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXEmbedders
import MLXFast
import Metal
import SwiftUI
import Tokenizers
import Combine
import CoreML
import PDFKit
import NaturalLanguage
import Hub


struct AskResponse: Decodable {
    let answer: String
}


@MainActor
class ChatViewModel: ObservableObject {
    @Published var input = ""
    @Published var finalContext = ""
    @Published var prompt = ""
    @Published var messages: [String] = []
    @Published private(set) var isReady = true
    @Published var isModelLoading: Bool = true
    @Published var isEmbedModelLoading: Bool = true
    @Published var modelLoadProgress: Progress? = nil
    @Published var embedModelProgress: Progress? = nil
    @Published var embedderModel: MLXEmbedders.ModelContainer?
    private var session: ChatSession?
    
    
    
    init() {
        Task {
            self.isModelLoading = true
            self.isEmbedModelLoading = true
            let progress = Progress(totalUnitCount: 100)
            let embedProgress = Progress(totalUnitCount: 100)
            self.modelLoadProgress = progress
            self.embedModelProgress = embedProgress
            do {
                let model = try await loadModel(id: "ShukraJaliya/BLUECOMPUTER.2", progressHandler: { [weak self] prog in
                    Task { @MainActor in
                        self?.modelLoadProgress = prog
                    }
                })
                self.session = ChatSession(model, generateParameters: .init(
                    maxTokens: 600,
                    temperature: 0.4,
                    topP: 0.9
                ))
            } catch {
                print("Model loading failed: \(error)")
            }
            
            self.isModelLoading = false
            
            do {
                let modelContainer = try await MLXEmbedders.loadModelContainer(configuration: ModelConfiguration.minilm_l6,  progressHandler: { [weak self] prog in
                    Task { @MainActor in
                        self?.embedModelProgress = prog
                    }
                })
                
                self.embedderModel = modelContainer
                
            } catch {
                print("Model loading failed: \(error)")
            }
            
            self.isEmbedModelLoading = false
            
        }
    }
    
    private func classifyTopic(for question: String) -> String? {
        guard let modelURL = Bundle.main.url(forResource: "TopicClassifier", withExtension: "mlmodelc") else { return nil }
        do {
            let model = try MLModel(contentsOf: modelURL)
            let input = try MLDictionaryFeatureProvider(dictionary: ["text": question])
            let prediction = try model.prediction(from: input)
            return prediction.featureValue(for: "label")?.stringValue
        } catch {
            print("Topic classification failed: \(error)")
            return nil
        }
    }
    
    private func textChunker(for question: String) -> [String] {
        guard let pdfFile = Bundle.main.url(forResource: "Final_Activity_v1", withExtension: "pdf"),
              let pdfDocument = PDFDocument(url: pdfFile) else {
            print("PDF not found")
            return []
        }
        
        var allText: [String] = []
        
        // STEP 1: Extract text from each page
        for pageIndex in 0..<pdfDocument.pageCount {
            if let page = pdfDocument.page(at: pageIndex),
               let pageText = page.string?.trimmingCharacters(in: .whitespacesAndNewlines),
               !pageText.isEmpty {
                allText.append(pageText)
            }
        }
        
        // STEP 2: Split into manageable chunks (like RecursiveCharacterTextSplitter)
        let chunkSize = 100
        let chunkOverlap = 5
        var chunks: [String] = []
        
        for text in allText {
            var start = text.startIndex
            while start < text.endIndex {
                let end = text.index(start, offsetBy: chunkSize, limitedBy: text.endIndex) ?? text.endIndex
                let chunk = String(text[start..<end])
                chunks.append(chunk)
                
                // Advance start by chunkSize - chunkOverlap
                start = text.index(start, offsetBy: chunkSize - chunkOverlap, limitedBy: text.endIndex) ?? text.endIndex
            }
        }
        
        return chunks
    }
    
    func embedChunks(_ chunks: [String]) async throws -> [[Float]] {
        guard let modelContainer = self.embedderModel else {
            throw NSError(domain: "Embedder", code: -1, userInfo: [NSLocalizedDescriptionKey: "Embedding model not loaded"])
        }
        
        return await modelContainer.perform { (model: EmbeddingModel, tokenizer, pooling) -> [[Float]] in
            let encodedInputs = chunks.map { text in
                tokenizer.encode(text: text, addSpecialTokens: true)
            }
            let maxLength = encodedInputs.map(\.count).max() ?? 0
            let eosTokenId = tokenizer.eosTokenId ?? 0
            let padded = stacked(
                encodedInputs.map { tokens in
                    MLXArray(tokens + Array(repeating: eosTokenId, count: maxLength - tokens.count))
                }
            )
            let mask = (padded .!= eosTokenId)
            let tokenTypes = MLXArray.zeros(like: padded)
            let output = pooling(
                model(padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
                normalize: true, applyLayerNorm: true
            )
            // Print shape for debugging
            print(output.shape)
            // Try to cast to [[Float]]
            if let embeddings = output.asArray(Float.self) as? [[Float]] {
                return embeddings
            } else {
                // Fallback: manually reshape
                let flat: [Float] = output.asArray(Float.self)
                let embeddingSize = flat.count / chunks.count
                return (0..<chunks.count).map { i in
                    Array(flat[i*embeddingSize..<(i+1)*embeddingSize])
                }
            }
        }
    }
    
    func retrieveContext(
        question: String,
        chunks: [String],
        chunkEmbeddings: [[Float]],
        topK: Int = 1
    ) async throws -> [String] {
        // Get the embedding for the question string.
        let questionEmbeddingArrs = try await embedChunks([question])
        guard let qEmb = questionEmbeddingArrs.first else { return [] }
        
        // Compute dot-product similarity to each chunk embedding.
        let similarities: [Float] = chunkEmbeddings.map { chunkEmb in
            dotProduct(qEmb, chunkEmb)
        }
        
        // Get indices of top-k values (descending)
        let topKIdx = similarities
            .enumerated()
            .sorted(by: { $0.element > $1.element })
            .prefix(topK)
            .map { $0.offset }
        
        // Return the corresponding chunk texts.
        return topKIdx.map { chunks[$0] }
    }
    
    /// Computes the dot product between two float arrays.
    private func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }
        return zip(a, b).map(*).reduce(0, +)
    }
    
    
    let SYSTEM_PROMPT = """
       You are an expert who only teaches data activism and Python programming to K–12 students. 
           You explain concepts step by step using clear, scaffolded language. 
           You never provide exact code solutions. 
           If a student submits code with question marks (?), explain what each line is supposed to do by guiding them with detailed conceptual steps. 
           For general programming questions (like "How to create a function?"), give a full explanation with a short example, but do not solve specific problems. 
           If a student asks something unrelated or off-topic, politely redirect them to focus on data activism or Python programming.
       """
    
    func send() {
        
        guard let session = self.session, !self.input.isEmpty else { return }
        let question = self.input
        self.messages.append("You: \(question)")
        self.input = ""
        self.isReady = false
        
        Task { @MainActor in
            let start = Date()
            do {
                
                if let topic = classifyTopic(for: question) {
                    print("Predicted topic: \(topic)")
                    
                    if topic == "1" {
                        let chunks = textChunker(for: question)
                        let chunkEmbeddings = try await embedChunks(chunks)
                        var topChunks = try await retrieveContext(
                            question: question,
                            chunks: chunks,
                            chunkEmbeddings: chunkEmbeddings,
                            topK: 1 // Change to more for more context
                        )
                        var contextText = topChunks.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
                        
                        if question.contains("?") && question.contains(":") {
                            contextText = ""
                        }
                        
                        // Avoid unhelpful or vague context and favor base model knowledge for general queries.
                        let loweredContext = contextText.lowercased().replacingOccurrences(of: "\n", with: "")
                        if contextText.count < 20 || ["e", ".", "", "for example:"].contains(loweredContext) {
                            contextText = ""
                        }
                        
                        if contextText.isEmpty {
                            self.finalContext = ""
                            prompt = """
                                     <|im_start|>system \(SYSTEM_PROMPT)<|im_end|>\
                                     <|im_start|>user \(question)<|im_end|>
                                     <|im_start|>assistant
                                     """
                        } else {
                            self.finalContext = contextText
                            prompt = """
                                     <|im_start|>system \(SYSTEM_PROMPT)<|im_end|>\
                                     <|im_start|>user \(question)
                                     \nContext:\n\(contextText)<|im_end|>
                                     <|im_start|>assistant 
                                     If the provided context is directly relevant, smoothly weave up to two supporting details from it into your explanation. Do not copy code or describe placeholder replacements unless the user pasted code with literal '?'.
                                    """
                        }
                    }
                }
                
                print("[Prompt sent to model]:\n\(prompt)")
                
                let userPrompt = prompt
                
                let reply = try await session.respond(to: userPrompt)
                let elapsed = Date().timeIntervalSince(start)
                self.messages.append("Bot (\(String(format: "%.2f", elapsed))s): \(reply)")
            } catch {
                let elapsed = Date().timeIntervalSince(start)
                self.messages.append("Error (\(String(format: "%.2f", elapsed))s): \(error.localizedDescription)")
            }
            self.isReady = true
            
        }
    }
    
}
