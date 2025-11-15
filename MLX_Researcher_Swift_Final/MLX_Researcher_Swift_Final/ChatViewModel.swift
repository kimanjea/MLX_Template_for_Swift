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
    // UI-facing state
    @Published var input = ""
    @Published var finalContext = ""
    @Published var prompt = ""
    @Published var messages: [String] = []
    @Published private(set) var isReady = true

    // Model loading
    @Published var isModelLoading: Bool = true
    @Published var isEmbedModelLoading: Bool = true
    @Published var modelLoadProgress: Progress? = nil
    @Published var embedModelProgress: Progress? = nil
    @Published var embedderModel: MLXEmbedders.ModelContainer?

    // RAG PDF selection: if nil, fall back to bundled Final_Activity_v1.pdf
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

            // Load main LLM
            do {
                let model = try await loadModel(
                    id: "ShukraJaliya/BLUECOMPUTER.2",
                    progressHandler: { [weak self] prog in
                        Task { @MainActor in
                            self?.modelLoadProgress = prog
                        }
                    }
                )

                self.session = ChatSession(
                    model,
                    instructions: SYSTEM_PROMPT,
                    generateParameters: GenerateParameters(
                        maxTokens: 600,
                        temperature: 0.4,
                        topP: 0.8
                    )
                )
            } catch {
                print("Model loading failed: \(error)")
            }

            self.isModelLoading = false

            // Load embedder model (for RAG + semantic helpers)
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

    // MARK: - Topic Classification

    private func classifyTopic(for question: String) -> String? {
        guard let modelURL = Bundle.main.url(forResource: "TopicClassifier", withExtension: "mlmodelc") else {
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

    // MARK: - PDF Chunking (RAG) with configurable PDF

    /// Uses either the currently selected RAG PDF or the bundled Final_Activity_v1.pdf,
    /// then performs semantic sentence-based chunking.
    private func textChunker(for question: String) -> [String] {
        let pdfURL: URL

        if let customURL = currentRAGPDFURL {
            pdfURL = customURL
        } else if let bundledURL = Bundle.main.url(forResource: "Final_Activity_v1", withExtension: "pdf") {
            pdfURL = bundledURL
        } else {
            print("PDF not found for RAG (neither uploaded nor bundled).")
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

    // MARK: - Embeddings + Retrieval

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
                model(
                    padded,
                    positionIds: nil,
                    tokenTypeIds: tokenTypes,
                    attentionMask: mask
                ),
                normalize: true,
                applyLayerNorm: true
            )
            print(output.shape)

            if let embeddings = output.asArray(Float.self) as? [[Float]] {
                return embeddings
            } else {
                let flat: [Float] = output.asArray(Float.self)
                let embeddingSize = flat.count / chunks.count
                return (0..<chunks.count).map { i in
                    Array(flat[i * embeddingSize..<(i + 1) * embeddingSize])
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

    // MARK: - System Prompt

    let SYSTEM_PROMPT = """
       You are an expert who only teaches data activism and Python programming to K–12 students. 
           You explain concepts step by step using clear, scaffolded language. 
           You never provide exact code solutions. 
           If a student submits code with question marks (?), explain what each line is supposed to do by guiding them with detailed conceptual steps. 
           For general programming questions (like "How to create a function?"), give a full explanation with a short example, but do not solve specific problems.  
           If a student asks something unrelated or off-topic, politely redirect them to focus on data activism or Python programming.
       """

    // MARK: - RAG PDF Setter (used by ContentView.uploadCourseView)

    func setRAGPDF(url: URL) {
        currentRAGPDFURL = url
        print("RAG PDF set to: \(url.path)")
    }

    // MARK: - Chat Send Logic (with topic gating + coding scaffold detection)

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

                    let isCodingScaffold =
                        question.contains("?") &&
                        (question.contains("def") || question.contains(":"))

                    if topic == "1" {
                        // On-topic
                        if isCodingScaffold {
                            // For scaffolded code, no RAG – just use the prompt
                            self.finalContext = ""
                            prompt = """
                            <|im_start|>system \(SYSTEM_PROMPT)<|im_end|>\
                            <|im_start|>user \(question)<|im_end|>
                            <|im_start|>assistant
                            """
                        } else {
                            // On-topic, not a code scaffold: use RAG from selected PDF
                            let chunks = textChunker(for: question)
                            let chunkEmbeddings = try await embedChunks(chunks)
                            let topChunks = try await retrieveContext(
                                question: question,
                                chunks: chunks,
                                chunkEmbeddings: chunkEmbeddings,
                                topK: 1
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
                        // Off-topic according to classifier – no RAG
                        self.finalContext = ""
                        prompt = """
                                 <|im_start|>system \(SYSTEM_PROMPT)<|im_end|>\
                                 <|im_start|>user \(question)<|im_end|>
                                 <|im_start|>assistant
                                 """
                    }
                }

                print("[Prompt sent to model]:\n\(prompt)")

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

    // MARK: - Extra PDF → JSONL Helpers (unchanged)

    /// Extracts the most common nouns/proper nouns from given texts using NLTagger
    private func extractKeywords(from texts: [String], maxKeywords: Int = 20) -> [String] {
        var frequency: [String: Int] = [:]
        for text in texts {
            let tagger = NLTagger(tagSchemes: [.lexicalClass])
            tagger.string = text
            tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .lexicalClass) { tag, range in
                if let tag = tag, tag == .noun {
                    let word = String(text[range]).lowercased()
                    if word.count > 2 {
                        frequency[word, default: 0] += 1
                    }
                }
                return true
            }
        }
        return frequency.sorted { $0.value > $1.value }
            .prefix(maxKeywords)
            .map { $0.key }
    }

    /// Finds the user question with the highest semantic similarity to the assistant response.
    private func semanticMatch(userQuestions: [String], assistantResponse: String) async throws -> String? {
        let embeddings = try await embedChunks(userQuestions + [assistantResponse])
        let userEmbeddings = embeddings.dropLast()
        let responseEmbedding = embeddings.last!
        let similarities = userEmbeddings.map { dotProduct($0, responseEmbedding) }
        if let maxIndex = similarities.enumerated().max(by: { $0.element < $1.element })?.offset {
            return userQuestions[maxIndex]
        }
        return nil
    }

    /// Cascading heuristic+semantic matching for user-assistant pairing
    func matchUserQuestionsToAssistantResponses(
        userQuestions: [String],
        assistantResponses: [String]
    ) async throws -> [(user: String, assistant: String)] {
        var usedQuestions = Set<String>()
        var results: [(user: String, assistant: String)] = []

        for response in assistantResponses {
            if let match = try await semanticMatch(
                userQuestions: userQuestions.filter { !usedQuestions.contains($0) },
                assistantResponse: response
            ) {
                if !usedQuestions.contains(match) {
                    results.append((user: match, assistant: response))
                    usedQuestions.insert(match)
                }
            }
        }
        return results
    }

    // Updated method with requested changes:
    func extractPDFToJsonLines(from url: URL) async {
        do {
            guard let document = PDFDocument(url: url) else {
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
            encoder.outputFormatting = []

            let examplePrompt = ConversationExample(
                system: "You are an expert who explains concepts step by step using clear, scaffolded language. You never provide exact code solutions. For questions with code or unclear elements, explain what each part means by guiding with detailed conceptual steps. For general questions (like 'How to..'), give a full explanation with a short example, but do not solve specific problems. If a user asks something off-topic, politely redirect them to focus on the relevant subject.",
                user: "USER INPUT HERE",
                assistant: "ASSISTANT RESPONSE HERE"
            )
            let systemPrompt: String = String(
                data: try encoder.encode(examplePrompt),
                encoding: .utf8
            )!

            let documentsDirs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
            guard let documentsDir = documentsDirs.first else {
                print("Failed to access Documents directory")
                return
            }

            let userQuestionsURL = documentsDir.appendingPathComponent("user_questions.txt")
            let userQuestions: [String]
            if let questionsData = try? Data(contentsOf: userQuestionsURL),
               let questionsString = String(data: questionsData, encoding: .utf8) {
                userQuestions = questionsString
                    .components(separatedBy: .newlines)
                    .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            } else {
                print("Warning: user_questions.txt not found or empty. Using placeholder questions.")
                userQuestions = ["USER INPUT HERE"]
            }

            let pairs = try await matchUserQuestionsToAssistantResponses(
                userQuestions: userQuestions,
                assistantResponses: lines
            )

            let examples = pairs.map {
                ConversationExample(system: systemPrompt, user: $0.user, assistant: $0.assistant)
            }

            let splitIndex = Int(Double(examples.count) * 0.8)
            let trainingExamples = examples[0..<splitIndex]
            let validExamples = examples[splitIndex...]

            let trainingURL = documentsDir.appendingPathComponent("training.jsonl")
            let validURL = documentsDir.appendingPathComponent("valid.jsonl")

            let trainingContent = trainingExamples.map {
                String(data: try! encoder.encode($0), encoding: .utf8)!
            }.joined(separator: "\n")
            let validContent = validExamples.map {
                String(data: try! encoder.encode($0), encoding: .utf8)!
            }.joined(separator: "\n")

            try trainingContent.write(to: trainingURL, atomically: true, encoding: .utf8)
            try validContent.write(to: validURL, atomically: true, encoding: .utf8)

            print("Training and validation files written to Documents directory in conversational prompt format.")
        } catch {
            print("Error extracting PDF to conversational prompt format: \(error)")
        }
    }

    /// Extracts likely user questions (sentences ending with '?') from a PDF file.
    func extractQuestionsFromPDF(url: URL) -> [String] {
        guard let document = PDFDocument(url: url) else {
            print("Failed to load PDF")
            return []
        }
        var allText = ""
        for pageIndex in 0..<document.pageCount {
            if let page = document.page(at: pageIndex),
               let pageText = page.string {
                allText += pageText + "\n"
            }
        }
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = allText
        var questions: [String] = []
        tokenizer.enumerateTokens(in: allText.startIndex..<allText.endIndex) { range, _ in
            let sentence = allText[range].trimmingCharacters(in: .whitespacesAndNewlines)
            if sentence.hasSuffix("?") {
                questions.append(sentence)
            }
            return true
        }
        return questions
    }

    /// Gathers user questions from PDF, optional user_questions.txt, plus fallback examples.
    func gatherUserQuestions(fromPDF url: URL) -> [String] {
        var questions = Set<String>()

        // 1. Extract from PDF
        let pdfQuestions = extractQuestionsFromPDF(url: url)
        questions.formUnion(pdfQuestions)

        // 2. Try to load from user_questions.txt
        let documentsDirs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        if let documentsDir = documentsDirs.first {
            let userQuestionsURL = documentsDir.appendingPathComponent("user_questions.txt")
            if let questionsData = try? Data(contentsOf: userQuestionsURL),
               let questionsString = String(data: questionsData, encoding: .utf8) {
                let txtQuestions = questionsString
                    .components(separatedBy: .newlines)
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                questions.formUnion(txtQuestions.filter { !$0.isEmpty })
            }
        }

        // 3. Filter out very short items
        let filtered = questions.filter { $0.count > 8 }
        var result = Array(filtered)

        // 4. Fallback/generic examples if list is too small
        if result.count < 5 {
            result.append(contentsOf: [
                "What does a variable do in Python?",
                "How can I use a for loop?",
                "What is data activism?",
                "How do I read a CSV file?",
                "What does it mean to analyze data?"
            ])
        }
        return Array(Set(result))
    }
}
