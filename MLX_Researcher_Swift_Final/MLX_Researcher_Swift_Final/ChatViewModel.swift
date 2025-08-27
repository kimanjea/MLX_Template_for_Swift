import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXEmbedders
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
    @Published var messages: [String] = []
    @Published private(set) var isReady = true
    @Published var isModelLoading: Bool = true
    @Published var isEmbedModelLoading: Bool = true
    @Published var modelLoadProgress: Progress? = nil
    @Published var embedModelProgress: Progress? = nil
    private var session: ChatSession?
    
    // Store PDF text chunks and their embeddings
    private var pdfChunks: [String] = []
    private var pdfEmbeddings: [[Float]] = []
    private var embedModelContainer: MLXEmbedders.ModelContainer?
    

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
                self.session = ChatSession(model, instructions: SYSTEM_PROMPT, generateParameters: GenerateParameters.init(temperature: 0.65,topP: 0.9 ))
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
                self.embedModelContainer = modelContainer
                
               
                
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
    
    /// Load a PDF from the given URL, extract text, split into chunks, embed them, and store embeddings.
    func loadAndEmbedPDF(pdfURL: URL) async {
        guard let embedModelContainer = self.embedModelContainer else {
            print("Embedding model is not loaded yet.")
            return
        }
        
        guard let document = PDFDocument(url: pdfURL) else {
            print("Failed to open PDF document")
            return
        }
        
        // Extract full text from PDF
        var fullText = ""
        for pageIndex in 0..<document.pageCount {
            guard let page = document.page(at: pageIndex),
                  let pageText = page.string else { continue }
            fullText.append(pageText + "\n")
        }
        
        // Split text into overlapping chunks (~800 chars with 50 char overlap)
        let chunkSize = 800
        let overlap = 50
        
        var chunks: [String] = []
        var start = fullText.startIndex
        
        while start < fullText.endIndex {
            let endIndex = fullText.index(start, offsetBy: chunkSize, limitedBy: fullText.endIndex) ?? fullText.endIndex
            let chunk = String(fullText[start..<endIndex])
            chunks.append(chunk)
            if endIndex == fullText.endIndex {
                break
            }
            start = fullText.index(endIndex, offsetBy: -overlap, limitedBy: fullText.startIndex) ?? fullText.startIndex
        }
        
        self.pdfChunks = chunks
        
        // Embed each chunk one by one using embedModelContainer (simple embedding without batching)
        var embeddings: [[Float]] = []
        for chunk in chunks {
            let emb = await embedModelContainer.perform { (model: EmbeddingModel, tokenizer, pooling) -> [Float] in
                let encoded = tokenizer.encode(text: chunk, addSpecialTokens: true)
                let input = MLXArray(encoded)
                let tokenTypes = MLXArray.zeros(like: input)
                let mask = (input .!= tokenizer.eosTokenId ?? 0)
                let embedding = pooling(
                    model(input, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
                    normalize: true, applyLayerNorm: true
                )
                return embedding.asArray(Float.self)
            }
            embeddings.append(emb)
        }
        self.pdfEmbeddings = embeddings
    }
    
    /// Retrieves context by embedding the question, finding closest chunks by cosine similarity, and returning topK chunks concatenated.
    func retrieveContext(for question: String, topK: Int) -> String {
        guard let embedModelContainer = self.embedModelContainer,
              !pdfChunks.isEmpty,
              !pdfEmbeddings.isEmpty else {
            return ""
        }
        
        var questionEmbedding: [Float] = []
        
        // Get embedding for question using simple embedding call (not batched)
        let semaphore = DispatchSemaphore(value: 0)
        Task {
            let emb = await embedModelContainer.perform { (model: EmbeddingModel, tokenizer, pooling) -> [Float] in
                let encoded = tokenizer.encode(text: question, addSpecialTokens: true)
                let input = MLXArray(encoded)
                let tokenTypes = MLXArray.zeros(like: input)
                let mask = (input .!= tokenizer.eosTokenId ?? 0)
                let embedding = pooling(
                    model(input, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
                    normalize: true, applyLayerNorm: true
                )
                return embedding.asArray(Float.self)
            }
            questionEmbedding = emb
            semaphore.signal()
        }
        semaphore.wait()
        
        // Compute cosine similarity between question embedding and each chunk embedding
        func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
            let dotProduct = zip(a, b).map(*).reduce(0, +)
            let magA = sqrt(a.map { $0 * $0 }.reduce(0, +))
            let magB = sqrt(b.map { $0 * $0 }.reduce(0, +))
            guard magA > 0 && magB > 0 else { return 0 }
            return dotProduct / (magA * magB)
        }
        
        let similarities = pdfEmbeddings.enumerated().map { (index, emb) in
            (index: index, similarity: cosineSimilarity(questionEmbedding, emb))
        }
        .sorted { $0.similarity > $1.similarity }
        
        // Select topK chunks with highest similarity
        let topChunks = similarities.prefix(topK).map { pdfChunks[$0.index] }
        
        return topChunks.joined(separator: "\n\n")
    }
    
    let SYSTEM_PROMPT = """
       You are an expert who only teaches data activism and Python programming to K–12 students.
       You explain concepts step by step using clear, scaffolded language.
       You never provide exact code solutions.
       If a student submits code with question marks (?), explain what each line is supposed to do by guiding them with detailed conceptual steps.
       For general programming questions (like "How do I create a function?"), give a detailed explanation with a short example, but do not solve specific student problems.
       """

    func send() {
        
        guard let session = self.session, !self.input.isEmpty else { return }
        let question = self.input
        self.messages.append("You: \(question)")
        self.input = ""
        self.isReady = false

        if let topic = classifyTopic(for: question) {
            print("Predicted topic: \(topic)")
            
            if topic == "1" {
                // Retrieve context from embedded PDF chunks if available
                self.finalContext = retrieveContext(for: question, topK: 1)
                print("Context: \(self.finalContext)")
            } else {
                print("Context should be nothing: \(self.finalContext)")
                self.finalContext = ""
            }
        }
        
        Task { @MainActor in
            let start = Date()
            do {
                
                let prompt = """
                <|im_start|>user
                 Answer the Question:
                \(question)
                <|im_end|>
                <|im_start|>assistant
                """
                
                let userPrompt = prompt
                let reply = try await session.respond(to: userPrompt)
                let elapsed = Date().timeIntervalSince(start)
                self.messages.append("(\(String(format: "%.2f", elapsed))s): \(reply)")
            } catch {
                let elapsed = Date().timeIntervalSince(start)
                self.messages.append("Error (\(String(format: "%.2f", elapsed))s): \(error.localizedDescription)")
            }
            self.isReady = true
        }
    }
}

