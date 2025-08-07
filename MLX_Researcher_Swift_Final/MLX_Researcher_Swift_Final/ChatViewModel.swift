import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Metal
import SwiftUI
import Tokenizers
import Combine

struct AskResponse: Decodable {
    let answer: String
}

@MainActor
class ChatViewModel: ObservableObject {
    @Published var input = ""
    @Published var messages: [String] = []
    @Published private(set) var isReady = true

    private let endpointURL = URL(string: "http://127.0.0.1:8000/ask")!

    init() {
        // Assume FastAPI server is running and ready
    }

    func send() {
        guard isReady,
              !input.trimmingCharacters(in: .whitespaces).isEmpty
        else { return }

        let question = input
        messages.append("You: \(question)")
        input = ""
        isReady = false
        Task { @MainActor in
            let start = Date()
            do {
                var request = URLRequest(url: endpointURL)
                request.timeoutInterval = 300
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                let payload = ["question": question]
                request.httpBody = try JSONEncoder().encode(payload)

                let (data, response) = try await URLSession.shared.data(for: request)
                guard let http = response as? HTTPURLResponse,
                      200..<300 ~= http.statusCode else {
                    throw URLError(.badServerResponse)
                }

                let askResp = try JSONDecoder().decode(AskResponse.self, from: data)
                let elapsed = Date().timeIntervalSince(start)
                messages.append("Bot (\(String(format: "%.2f", elapsed))s): \(askResp.answer)")
            } catch {
                let elapsed = Date().timeIntervalSince(start)
                messages.append("Error (\(String(format: "%.2f", elapsed))s): \(error.localizedDescription)")
            }
            isReady = true
        }
    }
}
