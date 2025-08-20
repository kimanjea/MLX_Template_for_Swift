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
import NaturalLanguage

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
    private var session: ChatSession?

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
    
    let SYSTEM_PROMPT = """
       You are an expert who only teaches data activism and Python programming to K–12 students.
       You explain concepts step by step using clear, scaffolded language.
       You never provide exact code solutions.
       If a student submits code with question marks (?), explain what each line is supposed to do by guiding them with detailed conceptual steps.
       For general programming questions (like "How do I create a function?"), give a detailed explanation with a short example, but do not solve specific student problems.
       """

    func send() {
        guard isReady,
              !input.trimmingCharacters(in: .whitespaces).isEmpty
        else { return }
        
        guard let session = session, !input.isEmpty else { return }
                
        let question = input
        messages.append("You: \(question)")
        input = ""
        isReady = false

        if let topic = classifyTopic(for: question) {
            print("Predicted topic: \(topic)")
            
            if topic == "1" {
                print("Context: \(finalContext)")
            } else {
                print("Context should be nothing: \(finalContext)")
                finalContext = ""
            }
        }
        
        Task { @MainActor in
            let start = Date()
            do {
                let elapsed = Date().timeIntervalSince(start)

                let prompt = """
                                <|im_start|>system
                                \(SYSTEM_PROMPT)
                                <|im_end|>
                                <|im_start|>user
                                 Answer the Question:
                                \(question)

                                <|im_end|>
                                <|im_start|>assistant
                                """
                
                let userPrompt = prompt
                let reply = try await session.respond(to: userPrompt)
                messages.append("(\(String(format: "%.2f", elapsed))s): \(reply)")
            } catch {
                let elapsed = Date().timeIntervalSince(start)
                messages.append("Error (\(String(format: "%.2f", elapsed))s): \(error.localizedDescription)")
            }
            isReady = true
        }
    }
}

