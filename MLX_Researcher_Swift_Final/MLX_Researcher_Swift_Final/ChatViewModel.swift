import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXEmbedders
import MLXFast
import MLXOptimizers
import Metal
import SwiftUI
import Tokenizers
import Combine
import CoreML
import PDFKit
import NaturalLanguage
import Hub

struct ConversationExample: Codable {
    let system: String
    let user: String
    let assistant: String
}

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
    @Published var modelLoadProgress: Foundation.Progress? = nil
    @Published var embedModelProgress: Foundation.Progress? = nil
    @Published var embedderModel: MLXEmbedders.ModelContainer?
    @Published var MinEmbedderModel: MLXEmbedders.ModelContainer?
    
    @Published var isTraining: Bool = false
    @Published var trainingProgress: Double? = nil
    @Published private var showSaveAdapterSheet = false
    @Published private var newAdapterName: String = ""
    @Published var savedAdapters: [String] = []
    private let adaptersDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Adapters", isDirectory: true)
    
    // New: keep references to cancel ongoing work
    private var modelLoadTask: Task<Void, Never>?
    private var embedderLoadTask: Task<Void, Never>?
    
    /// If nil, we fall back to Final_Activity_v1.pdf in the app bundle
    @Published var currentRAGPDFURL: URL? = nil

    private var session: ChatSession?
    
    // LoRA training configuration
    private let loraLayers = 4
    private let learningRate: Float = 1e-5
    private let trainingIterations: Int = 200
    @Published var didFinishExtraction: Bool = false
    @Published var isAdapterActive = false

    // Generation parameters for post-training evaluation (optional)
    private let generateTemperature: Float = 0.6
    private let generateTopP: Float = 0.9
    private let evaluateShowEvery = 8
    private let maxTokens = 200

    // Cache the loaded model container for training
    private var modelContainerForTraining: MLXLMCommon.ModelContainer?
    

    
    init() {
        Task {
            self.isModelLoading = true
            self.isEmbedModelLoading = true
            let progress = Foundation.Progress(totalUnitCount: 100)
            let embedProgress = Foundation.Progress(totalUnitCount: 100)
            self.modelLoadProgress = progress
            self.embedModelProgress = embedProgress
            
            // 1) Initial model load (cancelable)
            modelLoadTask?.cancel()
            modelLoadTask = Task { [currentModelID] in
                await performModelLoad(for: currentModelID)
            }
            
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
    
    private func performModelLoad(for modelID: String) async {
        // Reset state on main actor (we're already @MainActor)
        isModelLoading = true
        isReady = false
        modelLoadProgress = Foundation.Progress(totalUnitCount: 100)

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
        // Prefer the user-selected PDF if available, otherwise fall back to the bundled PDF
        let pdfURL: URL?
        if let current = self.currentRAGPDFURL {
            pdfURL = current
        } else {
            pdfURL = Bundle.main.url(forResource: "Final_Activity_v1", withExtension: "pdf")
        }

        guard let pdfURL, let pdfDocument = PDFDocument(url: pdfURL) else {
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
                                     <|im_start|>system \(SYSTEM_PROMPT).<|im_end|>
                                     <|im_start|>user \(question)<|im_end|>
                                     <|im_start|>assistant
                                     """
                        }
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
    

    
    // Updated method with requested changes:
    func extractPDFToJsonLines(from url: URL) async {
        // Prefer the provided URL; if nil (shouldn't happen), fall back to the currentRAGPDFURL.
        let effectiveURL = url
        

        do {
            // 1. Load PDF and extract all text
            guard let document = PDFDocument(url: effectiveURL) else {
                print("Failed to load PDF")
                return
            }
            
            var allText = ""
            for pageIndex in 0..<document.pageCount {
                if let page = document.page(at: pageIndex),
                   let pageText = page.string {
                    allText += pageText + "\n"
                }
            }
            
            // 2. Split into sentences using NLTokenizer
            let tokenizer = NLTokenizer(unit: .sentence)
            tokenizer.string = allText
            var lines: [String] = []
            tokenizer.enumerateTokens(in: allText.startIndex..<allText.endIndex) { range, _ in
                let sentence = allText[range].trimmingCharacters(in: .whitespacesAndNewlines)
                if !sentence.isEmpty {
                    lines.append(sentence)
                }
                return true
            }
            
            let encoder = JSONEncoder()
            
            // 3. Plain system prompt string (no double-encoding)
            let systemPrompt = SYSTEM_PROMPT2
            
            // 4. Use documents directory in user domain
            let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            print("Writing dataset files to:", documentsDir.path)

            
            let jsonlLines = lines.map { sentence -> String in
                let dict = ["text": "Instruction: \(systemPrompt)\nAssistant: \(sentence)"]
                let data = try! encoder.encode(dict)
                return String(data: data, encoding: .utf8)!
            }
            
            
            // 8. Split into train / valid (80 / 20)
            let splitIndex = Int(Double(jsonlLines.count) * 0.8)
            let trainingLines = jsonlLines[..<splitIndex]
            let validLines = jsonlLines[splitIndex...]
            
            let trainingURL = documentsDir.appendingPathComponent("train.jsonl")
            let validURL    = documentsDir.appendingPathComponent("valid.jsonl")
            

            // 9. Append to train.jsonl (newline-safe)
            let trainingContent = trainingLines.joined(separator: "\n")
            if FileManager.default.fileExists(atPath: trainingURL.path),
               let handle = try? FileHandle(forUpdating: trainingURL) {
                defer { try? handle.close() }
                // Read last byte to see if file already ends with a newline
                let fileSize = (try? FileManager.default.attributesOfItem(atPath: trainingURL.path)[.size] as? NSNumber)?.intValue ?? 0
                handle.seek(toFileOffset: fileSize > 0 ? UInt64(fileSize - 1) : 0)
                let lastByte = fileSize > 0 ? handle.readData(ofLength: 1) : Data()
                handle.seekToEndOfFile()
                if let lastChar = String(data: lastByte, encoding: .utf8), lastChar != "\n", !trainingContent.isEmpty {
                    handle.write("\n".data(using: .utf8)!)
                }
                if let data = trainingContent.data(using: .utf8) {
                    handle.write(data)
                    // Do not add trailing newline; assume content already has correct line endings
                }
            } else {
                try trainingContent.write(to: trainingURL, atomically: true, encoding: .utf8)
            }

            // 10. Append to valid.jsonl (newline-safe)
            let validContent = validLines.joined(separator: "\n")
            if FileManager.default.fileExists(atPath: validURL.path),
               let handle = try? FileHandle(forUpdating: validURL) {
                defer { try? handle.close() }
                let fileSize = (try? FileManager.default.attributesOfItem(atPath: validURL.path)[.size] as? NSNumber)?.intValue ?? 0
                handle.seek(toFileOffset: fileSize > 0 ? UInt64(fileSize - 1) : 0)
                let lastByte = fileSize > 0 ? handle.readData(ofLength: 1) : Data()
                handle.seekToEndOfFile()
                if let lastChar = String(data: lastByte, encoding: .utf8), lastChar != "\n", !validContent.isEmpty {
                    handle.write("\n".data(using: .utf8)!)
                }
                if let data = validContent.data(using: .utf8) {
                    handle.write(data)
                }
            } else {
                try validContent.write(to: validURL, atomically: true, encoding: .utf8)
            }
            
            print("Training and validation files written to \(documentsDir.path) in conversational prompt format.")
            await MainActor.run {
                self.didFinishExtraction = true
            }

        } catch {
            print("Error extracting PDF to conversational prompt format: \(error)")
        }
    }
    

    // MARK: - Training Orchestration
    func trainFromCurrentPDF() {
        guard let url = currentRAGPDFURL else {
            print("No current PDF set. Did you call vm.setRAGPDF(url:)?")
            return
        }
        Task {
            await trainFromPDF(url: url)
        }
    }

    private func trainFromPDF(url: URL) async {
        await MainActor.run {
            self.isTraining = true
            self.trainingProgress = nil
            self.didFinishExtraction = false
        }

        // 1) Extract dataset files from the PDF (train/valid)
        await extractPDFToJsonLines(from: url)

        // 2) Determine the dataset URLs (must match extractPDFToJsonLines output)
        let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let trainURL = documentsDir.appendingPathComponent("train.jsonl")
        let validURL = documentsDir.appendingPathComponent("valid.jsonl")
        let testURL  = documentsDir.appendingPathComponent("test.jsonl")
        

        // 3) Run LoRA training locally
        do {
            try await trainLocallyWithLoRA(trainURL: trainURL, validURL: validURL, testURL: testURL)
        } catch {
            print("Training failed: \(error)")
        }

        await MainActor.run {
            self.isTraining = false
            self.trainingProgress = nil
        }
    }

    // MARK: - LoRA Training Core
    private func loadTrainingModelContainer() async throws -> MLXLMCommon.ModelContainer {
        let modelID = "Qwen/Qwen2.5-1.5B-Instruct"
        let config = LLMModelFactory.shared.configuration(id: modelID)
        let container = try await LLMModelFactory.shared.loadContainer(
            hub: HubApi(), configuration: config
        ) { prog in
            Task { @MainActor in
                self.modelLoadProgress = prog
            }
        }
        return container
    }

    private func loadLoRAData(from url: URL) throws -> [String] {
        return try MLXLLM.loadLoRAData(url: url)
    }

    private func trainLocallyWithLoRA(trainURL: URL, validURL: URL, testURL: URL?) async throws {
        GPU.set(cacheLimit: 32 * 1024 * 1024)

        let modelContainer: MLXLMCommon.ModelContainer
        if let cached = modelContainerForTraining {
            modelContainer = cached
        } else {
            modelContainer = try await loadTrainingModelContainer()
            modelContainerForTraining = modelContainer
        }

        let _ = try await modelContainer.perform { context in
            try LoRAContainer.from(
                model: context.model,
                configuration: LoRAConfiguration(numLayers: loraLayers)
            )
        }

        let train = try loadLoRAData(from: trainURL)
        let valid = try loadLoRAData(from: validURL)
        print("Train examples: \(train.count), Valid examples: \(valid.count)")

        try await modelContainer.perform { context in
            let optimizer = Adam(learningRate: self.learningRate)
            let params = LoRATrain.Parameters(batchSize: 1, iterations: self.trainingIterations)

            try LoRATrain.train(
                model: context.model,
                train: train,
                validate: valid,
                optimizer: optimizer,
                tokenizer: context.tokenizer,
                parameters: params
            ) { progress in
                Task { @MainActor in
                    switch progress {
                    case .train(let i, _, _, _):
                        print("LoRA iteration \(i)")
                        self.trainingProgress = Double(i) / Double(self.trainingIterations)
                    case .validation:
                        break
                    default:
                        break
                    }
                }
                return .more
            }
        }

        if let testURL, FileManager.default.fileExists(atPath: testURL.path) {
            let test = try loadLoRAData(from: testURL)
            let loss = await modelContainer.perform { context in
                LoRATrain.evaluate(
                    model: context.model,
                    dataset: test,
                    tokenizer: context.tokenizer,
                    batchSize: 1,
                    batchCount: 0
                )
            }
            await MainActor.run {
                self.messages.append("Training complete. Test loss \(loss.formatted()), ppl \(exp(loss).formatted())")
            }
        } else {
            await MainActor.run {
                self.messages.append("Training complete.")
            }
        }
    }
    
    func reloadSavedAdapters() {
        var names: [String] = []
        if FileManager.default.fileExists(atPath: adaptersDir.path) {
            if let contents = try? FileManager.default.contentsOfDirectory(at: adaptersDir, includingPropertiesForKeys: nil) {
                names = contents.filter { $0.hasDirectoryPath }.map { $0.lastPathComponent }
            }
        }
        self.savedAdapters = names.sorted()
    }
    nonisolated func saveLoRAAdapters(from model: LoRAContainer, to directory: URL) throws {
        // TODO: real save logic; for now, maybe just ensure dir exists
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // No-op for now
    }
    
    nonisolated func loadLoRAAdapters(into lora: LoRAContainer, from directory: URL) throws {
        // TODO: implement actual load logic for your MLX version
        // For now, ensure directory exists; real implementation should read adapter weights and load into `lora`.
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw NSError(domain: "LoRA", code: -2, userInfo: [NSLocalizedDescriptionKey: "Adapter directory not found at \(directory.path)"])
        }
    }

    // Call after training completes, passing a user-provided name.
    func saveCurrentAdapters(named name: String) async {
        // Ensure directory exists
        try? FileManager.default.createDirectory(at: adaptersDir, withIntermediateDirectories: true)
        

        let targetDir = adaptersDir.appendingPathComponent(name, isDirectory: true)
        // Remove existing with same name
        if FileManager.default.fileExists(atPath: targetDir.path) {
            try? FileManager.default.removeItem(at: targetDir)
        }
        try? FileManager.default.createDirectory(at: targetDir, withIntermediateDirectories: true)

        // Persist LoRA weights from the training container.
        // Uses MLX LoRA APIs to export adapters from the model used during training.
        if let modelContainer = modelContainerForTraining {
            do {
                try await modelContainer.perform { context in
                    let lora = try LoRAContainer.from(
                        model: context.model,
                        configuration: LoRAConfiguration(numLayers: self.loraLayers)
                    )
                    try saveLoRAAdapters(from: lora, to: targetDir)
                }
            } catch {
                print("Failed to save adapters: \(error)")
            }
        }

        // Write a small manifest for bookkeeping (non-fatal if it fails)
        let manifest: [String: Any] = [
            "name": name,
            "date": ISO8601DateFormatter().string(from: Date()),
            "baseModelID": "Qwen/Qwen2.5-1.5B-Instruct",
            "loraLayers": loraLayers,
            "learningRate": learningRate,
            "trainingIterations": trainingIterations
        ]
        if let data = try? JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted]) {
            let manifestURL = targetDir.appendingPathComponent("manifest.json")
            try? data.write(to: manifestURL)
        }

        await MainActor.run {
            self.reloadSavedAdapters()
            print("Saved adapters to \(targetDir.path)")
        }
    }

    // Apply a saved adapter to the current chat inference session’s model
    func applyAdapter(named name: String) async {
        let dir = adaptersDir.appendingPathComponent(name, isDirectory: true)
        guard FileManager.default.fileExists(atPath: dir.path) else {
            print("Adapter not found at \(dir.path)")
            return
        }

        do {
            let baseID = "ShukraJaliya/general"
            let model = try await loadModel(id: baseID, progressHandler: { _ in })

            // 2) Attach LoRA container
            let lora = try LoRAContainer.from(
                model: model.model,
                configuration: LoRAConfiguration(numLayers: self.loraLayers)
            )

            // 3) Load adapters from disk into the LoRA container
            try loadLoRAAdapters(into: lora, from: dir)

            // 4) Recreate chat session with adapted model
            self.session = ChatSession(
                model,
                instructions: SYSTEM_PROMPT2,
                generateParameters: GenerateParameters(maxTokens: 600, temperature: 0.4, topP: 0.8)
            )
            self.isAdapterActive = true

            print("Applied adapter: \(name)")
            self.messages.append("Applied adapter: \(name)")
        } catch {
            print("Failed to apply adapter: \(error)")
            self.messages.append("Failed to apply adapter: \(error.localizedDescription)")
        }
    }
    
    /// Delete a saved LoRA adapter directory and update the in-memory list.
    func deleteAdapter(named name: String) {
        let dir = adaptersDir.appendingPathComponent(name, isDirectory: true)

        do {
            if FileManager.default.fileExists(atPath: dir.path) {
                try FileManager.default.removeItem(at: dir)
                print("Deleted adapter at \(dir.path)")
            } else {
                print("Adapter directory not found at \(dir.path)")
            }

            // Remove from the in-memory list
            if let idx = savedAdapters.firstIndex(of: name) {
                savedAdapters.remove(at: idx)
            }

            // If this adapter was active, mark it inactive.
            if isAdapterActive {
                // You can also choose to reload a base model here if you want.
                isAdapterActive = false
            }
        } catch {
            print("Failed to delete adapter '\(name)': \(error)")
        }
    }
}

