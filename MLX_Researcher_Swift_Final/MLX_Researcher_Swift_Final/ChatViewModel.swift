import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Metal
import SwiftUI
import Tokenizers
import Combine
import CoreML
import PDFKit

struct AskResponse: Decodable {
    let answer: String
}

@MainActor
class ChatViewModel: ObservableObject {
    @Published var input = ""
    @Published var messages: [String] = []
    @Published private(set) var isReady = true
    @Published var isModelLoading: Bool = true
    private var session: ChatSession?

    private let endpointURL = URL(string: "http://127.0.0.1:8000/ask")!

    init() {
        Task {
            self.isModelLoading = true
            do {
                let model = try await loadModel(id: "ShukraJaliya/BLUECOMPUTER.2")
                session = ChatSession(model)
            } catch {
                print("Model loading failed: \(error)")
            }
            self.isModelLoading = false
        }
    }
    
    func extractTextFromPDF(named pdfFileName: String) -> String? {
        guard let pdfURL = Bundle.main.url(forResource: pdfFileName, withExtension: "pdf"),
              let pdfDocument = PDFDocument(url: pdfURL) else {
            print("Failed to load PDF")
            return nil
        }
        
        var fullText = ""
        for pageIndex in 0..<pdfDocument.pageCount {
            guard let page = pdfDocument.page(at: pageIndex) else { continue }
            if let pageText = page.string {
                fullText += pageText + "\n"
            }
        }
        return fullText
    }
    
    func splitTextIntoChunks(_ text: String, chunkSize: Int = 250, overlap: Int = 50) -> [String] {
        let words = text.components(separatedBy: .whitespacesAndNewlines)
        var chunks: [String] = []
        var index = 0
        
        while index < words.count {
            let end = min(index + chunkSize, words.count)
            let chunk = words[index..<end].joined(separator: " ")
            chunks.append(chunk)
            index += chunkSize - overlap
        }
        return chunks
    }
    
    func retrieveContextChunks(for question: String, topK: Int = 1) -> [String] {
        guard let pdfText = extractTextFromPDF(named: "Final_Activity_v1") else { return [] }
        let chunks = splitTextIntoChunks(pdfText)
        // Simple relevance: chunks containing the question substring
        let relevantChunks = chunks.filter { $0.localizedCaseInsensitiveContains(question) }
        if !relevantChunks.isEmpty {
            return Array(relevantChunks.prefix(topK))
        } else {
            return Array(chunks.prefix(topK)) // Fallback: just take the first chunk(s)
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

    func send() {
        guard isReady,
              !input.trimmingCharacters(in: .whitespaces).isEmpty
        else { return }
        
        guard let session = session, !input.isEmpty else { return }

        let question = input
        let contextChunks = retrieveContextChunks(for: question, topK: 1)
        let context = contextChunks.joined(separator: "\n")

        if let topic = classifyTopic(for: question) {
            print("Predicted topic: \(topic)")
            print("Context: \(context)")
        }
        messages.append("You: \(question)")
        input = ""
        isReady = false
        
        Task { @MainActor in
            let start = Date()
            do {
                let elapsed = Date().timeIntervalSince(start)
                let userPrompt = question
                let reply = try await session.respond(to: userPrompt)
                messages.append("Bot (\(String(format: "%.2f", elapsed))s): \(reply)")
            } catch {
                let elapsed = Date().timeIntervalSince(start)
                messages.append("Error (\(String(format: "%.2f", elapsed))s): \(error.localizedDescription)")
            }
            isReady = true
        }
    }
}

