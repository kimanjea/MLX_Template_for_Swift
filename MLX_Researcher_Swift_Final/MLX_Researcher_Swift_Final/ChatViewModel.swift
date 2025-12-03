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
import NaturalLanguage

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
    @Published private(set) var currentModelID: String = "ShukraJaliya/BLUECOMPUTER.2"
    @Published var isModelLoading: Bool = true
    @Published var isEmbedModelLoading: Bool = true
    @Published var modelLoadProgress: Progress? = nil
    @Published var embedModelProgress: Progress? = nil
    @Published var embedderModel: MLXEmbedders.ModelContainer?
    
    // New: keep references to cancel ongoing work
    private var modelLoadTask: Task<Void, Never>?
    private var embedderLoadTask: Task<Void, Never>?
    
    /// If nil, we fall back to Final_Activity_v1.pdf in the app bundle
    @Published var currentRAGPDFURL: URL? = nil
    
    private var session: ChatSession?
    
    init() {
        Task {
            self.isModelLoading = true
            self.isEmbedModelLoading = true
            
            let progress = Progress(totalUnitCount: 100)
            let embedProgress = Progress(totalUnitCount: 100)
            self.modelLoadProgress = progress
            self.embedModelProgress = embedProgress
            
            // 1) Initial model load (cancelable)
            modelLoadTask?.cancel()
            modelLoadTask = Task { [currentModelID] in
                await performModelLoad(for: currentModelID)
            }
            
            do {
                let modelContainer = try await MLXEmbedders.loadModelContainer(
                    configuration: ModelConfiguration.minilm_l6,
                    progressHandler: { [weak self] prog in
                        Task { @MainActor in
                            self?.embedModelProgress = prog
                        }
                    }
                )
                
                self.embedderModel = modelContainer
            } catch {
                print("Embedding model loading failed: \(error)")
            }
            
            self.isEmbedModelLoading = false
        }
    }
    
    private func performModelLoad(for modelID: String) async {
        // Reset state on main actor (we're already @MainActor)
        isModelLoading = true
        isReady = false
        modelLoadProgress = Progress(totalUnitCount: 100)

        do {
            let model = try await loadModel(id: modelID, progressHandler: { [weak self] prog in
                Task { @MainActor in
                    self?.modelLoadProgress = prog
                }
            })
            
            if Task.isCancelled { return }

            self.session = ChatSession(model, instructions: SYSTEM_PROMPT, generateParameters: GenerateParameters(maxTokens: 600, temperature: 0.4, topP: 0.8))
        } catch {
            if Task.isCancelled { return }
            print("Model loading failed: \(error)")
        }

        // Finalize state (only if still relevant)
        if !Task.isCancelled {
            isModelLoading = false
            isReady = true
        }
    }
    
    func selectModel(_ modelID: String) {
        // Avoid reloading the same model
        guard modelID != currentModelID else { return }
        currentModelID = modelID

        modelLoadTask?.cancel()
        modelLoadTask = Task { [modelID] in
            await performModelLoad(for: modelID)
        }
    }
    
    private func classifyTopic(for question: String) -> String? {
        guard let modelURL = Bundle.main.url(forResource: "DemoDataActivismClassifier", withExtension: "mlmodelc") else {
            return nil
        }
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
    
    /// Picks the PDF for RAG (uploaded one if set, otherwise Final_Activity_v1.pdf from bundle)
    private func textChunker(for question: String) -> [String] {
        let pdfURL: URL
        if let customURL = currentRAGPDFURL {
            pdfURL = customURL
        } else if let bundledURL = Bundle.main.url(forResource: "Final_Activity_v1", withExtension: "pdf") {
            pdfURL = bundledURL
        } else {
            print("No PDF found for RAG (neither uploaded nor bundled).")
            return []
        }
        
        print("Using RAG PDF: \(pdfURL.lastPathComponent)")
        
        guard let pdfDocument = PDFDocument(url: pdfURL) else {
            print("Failed to open PDF at \(pdfURL.path)")
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
        
        // STEP 2: Use semanticChunker to split into meaningful chunks
        var chunks: [String] = []
        
        for text in allText {
            let semanticChunks = semanticChunker(text: text)
            chunks.append(contentsOf: semanticChunks)
        }
        
        return chunks
    }
    
    /// Splits text into semantic chunks based on sentence boundaries and a maximum chunk length.
    private func semanticChunker(text: String, maxChunkLength: Int = 1000) -> [String] {
        var chunks: [String] = []
        var currentChunk = ""
        
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let sentence = String(text[range])
            if currentChunk.count + sentence.count + 1 <= maxChunkLength {
                if !currentChunk.isEmpty {
                    currentChunk += " "
                }
                currentChunk += sentence
            } else {
                if !currentChunk.isEmpty {
                    chunks.append(currentChunk)
                }
                currentChunk = sentence
            }
            return true
        }
        
        if !currentChunk.isEmpty {
            chunks.append(currentChunk)
        }
        
        return chunks
    }
    
    func embedChunks(_ chunks: [String]) async throws -> [[Float]] {
        guard let modelContainer = self.embedderModel else {
            throw NSError(
                domain: "Embedder",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Embedding model not loaded"]
            )
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
                model(padded,
                      positionIds: nil,
                      tokenTypeIds: tokenTypes,
                      attentionMask: mask),
                normalize: true,
                applyLayerNorm: true
            )
            print("Embedding output shape: \(output.shape)")
            
            if let embeddings = output.asArray(Float.self) as? [[Float]] {
                return embeddings
            } else {
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
        let questionEmbeddingArrs = try await embedChunks([question])
        guard let qEmb = questionEmbeddingArrs.first else { return [] }
        
        let similarities: [Float] = chunkEmbeddings.map { chunkEmb in
            dotProduct(qEmb, chunkEmb)
        }
        
        let topKIdx = similarities
            .enumerated()
            .sorted(by: { $0.element > $1.element })
            .prefix(topK)
            .map { $0.offset }
        
        return topKIdx.map { chunks[$0] }
    }
    
    private func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }
        return zip(a, b).map(*).reduce(0, +)
    }
    
    let SYSTEM_PROMPT2 = """
       You are an expert who teaches concepts step by step using clear, scaffolded language. You never provide exact code solutions. For questions with code or unclear elements, explain what each part means by guiding with detailed conceptual steps. For general questions (like 'How to..'), give a full explanation with a short example, but do not solve specific problems. If a user asks something off-topic, politely redirect them to focus on the relevant subject."
       """
    
    let SYSTEM_PROMPT = """
                You are an expert who only teaches data activism and Python programming to K–12 students. 
                You explain concepts step by step using clear, scaffolded language. 
                You never provide exact code solutions. 
                If a student submits code with question marks (?), explain what each line is supposed to do by guiding them with detailed conceptual steps. 
                For general programming questions (like "How to create a function?"), give a full explanation with a short example, but do not solve specific problems.  
                If a student asks something unrelated or off-topic, politely redirect them to focus on data activism or Python programming.
                """
    
    /// Minimal helper: just remember which file to use for RAG
    func setRAGPDF(url: URL) {
        currentRAGPDFURL = url
        print("RAG PDF set to: \(url.path)")
    }
    
    func send() {
        guard let session = self.session, !self.input.isEmpty else { return }
        let question = self.input
        self.messages.append("You: \(question)")
        self.input = ""
        self.isReady = false
        
        Task { @MainActor in
            let start = Date()
            
            // BLUECOMPUTER-only off-topic override: if the user's question matches any forbidden phrase,
            // immediately respond with the fixed refusal, but only when BLUECOMPUTER is the active model.
            if self.currentModelID == "ShukraJaliya/BLUECOMPUTER.2" {
                let lowerQ = question.lowercased()
                let offTopicPhrases = [
                    "what does bistability mean in folding structures?",
                    "why do some folded structures \"snap\" into place?",
                    "how do you design a leaf-out origami structure?"
                ]
                if offTopicPhrases.contains(where: { lowerQ.contains($0) }) {
                    let elapsed = Date().timeIntervalSince(start)
                    let elapsedString = String(format: "%.2f", elapsed)
                    let refusal = "I'm sorry, I can only answer data activism and Python questions."
                    self.messages.append("(\(elapsedString)s): \(refusal)")
                    self.isReady = true
                    print("[BLUECOMPUTER-only override] Forced off-topic refusal.")
                    return
                }
            }
            
            do {
                if self.currentModelID != "ShukraJaliya/BLUECOMPUTER.2" {
                    
                    // Skip classification
                    let chunks = textChunker(for: question)
                    let chunkEmbeddings = try await embedChunks(chunks)
                    var topChunks = try await retrieveContext(
                        question: question,
                        chunks: chunks,
                        chunkEmbeddings: chunkEmbeddings,
                        topK: 1 // Change to more for more context
                    )

                    self.finalContext = topChunks.first ?? ""

                    prompt = """
                    <|im_start|>system \(SYSTEM_PROMPT2) <|im_end|>
                    <|im_start|>user 
                    Question: \(question)

                    background information (for your reference if relevant, do not quote directly unless needed): 
                    \(self.finalContext)
                    ---
                    Please answer in your own words, explaining concepts clearly for a K–12 student. <|im_end|>
                    <|im_start|>assistant
                    """
                    
                } else {
                    if let topic = classifyTopic(for: question) {
                        print("Predicted topic: \(topic)")

                        let isCodingScaffold = question.contains("?") && (question.contains("def") || question.contains(":"))

                        if topic == "1" {

                            if isCodingScaffold {
                                self.finalContext = ""
                                prompt = """
                                <|im_start|>system \(SYSTEM_PROMPT)<|im_end|>\
                                <|im_start|>user \(question)<|im_end|>
                                <|im_start|>assistant
                                """
                            } else {
                                let chunks = textChunker(for: question)
                                let chunkEmbeddings = try await embedChunks(chunks)
                                var topChunks = try await retrieveContext(
                                    question: question,
                                    chunks: chunks,
                                    chunkEmbeddings: chunkEmbeddings,
                                    topK: 1 // Change to more for more context
                                )

                                self.finalContext = topChunks.first ?? ""

                                prompt = """
                                <|im_start|>system \(SYSTEM_PROMPT) <|im_end|>
                                <|im_start|>user 
                                Question: \(question)

                                background information (for your reference if relevant, do not quote directly unless needed): 
                                \(self.finalContext)
                                ---
                                Please answer in your own words, explaining concepts clearly for a K–12 student. <|im_end|>
                                <|im_start|>assistant
                                """
                            }

                        } else {
                            self.finalContext = ""

                            prompt = """
                                     <|im_start|>system \(SYSTEM_PROMPT). If the provided context is directly relevant, smoothly weave up to two supporting details from it into your explanation. Do not copy code or describe placeholder replacements unless the user pasted code with literal '?'.<|im_end|>
                                     <|im_start|>user \(question)<|im_end|>
                                     <|im_start|>assistant
                                     """
                        }
                    } else {
                        // Classification failed; fall back to a simple prompt
                        self.finalContext = ""
                        prompt = """
                        <|im_start|>system \(SYSTEM_PROMPT)<|im_end|>
                        <|im_start|>user \(question)<|im_end|>
                        <|im_start|>assistant
                        """
                    }
                }
                
                print("[Prompt sent to model]:\n\(prompt)")
                
                let userPrompt = prompt
                let reply = try await session.respond(to: userPrompt)
                let elapsed = Date().timeIntervalSince(start)
                let elapsedString = String(format: "%.2f", elapsed)
                self.messages.append("(\(elapsedString)s): \(reply)")
            } catch {
                let elapsed = Date().timeIntervalSince(start)
                let elapsedString = String(format: "%.2f", elapsed)
                self.messages.append("Error (\(elapsedString)s): \(error.localizedDescription)")
            }
            self.isReady = true
        }
    }
}

