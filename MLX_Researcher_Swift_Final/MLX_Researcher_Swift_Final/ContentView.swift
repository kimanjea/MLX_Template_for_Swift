import SwiftUI

struct ContentView: View {
    @StateObject private var vm = ChatViewModel()

    // Your static suggestions
    let promptSuggestions = [
        "What is data activism?",
        "Who are important figures in data activism?",
        "How can I become a data activist?",
        "Explain a Python function",
        "Show a pie-chart example",
        "What is black-box programming?",
        "Quiz me on data terms"
    ]

    var body: some View {
        NavigationView {
            HStack(spacing: 0) {
                // MARK: – Sidebar
                VStack(alignment: .leading) {
                    Text("Suggested Questions")
                        .font(.headline)
                        .padding(.top, 16)
                        .padding(.horizontal)

                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(promptSuggestions, id: \.self) { suggestion in
                                Text(suggestion)
                                    .font(.subheadline)
                                    .padding(.vertical, 8)
                                    .padding(.horizontal, 12)
                                    .background(
                                        Capsule()
                                            .stroke(Color.gray.opacity(0.6), lineWidth: 1)
                                    )
                                    .foregroundColor(.primary)
                                    // no .onTapGesture -> purely decorative
                            }
                        }
                        .padding(.vertical)
                    }
                }
                .frame(minWidth: 200, maxWidth: 250)
        

                Divider()

                // MARK: – Chat area
                VStack {
                    ScrollViewReader { proxy in
                        ScrollView {
                            VStack(alignment: .leading, spacing: 12) {
                                ForEach(vm.messages.indices, id: \.self) { idx in
                                    let msg = vm.messages[idx]
                                    HStack {
                                        if msg.starts(with: "You:") { Spacer() }

                                        Text(msg
                                            .replacingOccurrences(of: "You: ", with: "")
                                            .replacingOccurrences(of: "Bot: ", with: "")
                                        )
                                        .padding(10)
                                        .background(
                                            RoundedRectangle(cornerRadius: 8)
                                                .fill(msg.starts(with: "You:") ?
                                                      Color.blue.opacity(0.1) :
                                                      Color.gray.opacity(0.1))
                                        )
                                        .frame(maxWidth: .infinity,
                                               alignment: msg.starts(with: "You:") ? .trailing : .leading)

                                        if !msg.starts(with: "You:") { Spacer() }
                                    }
                                    .id(idx)
                                }
                            }
                            .padding()
                        }
                        .onChange(of: vm.messages.count) { _ in
                            if let last = vm.messages.indices.last {
                                withAnimation {
                                    proxy.scrollTo(last, anchor: .bottom)
                                }
                            }
                        }
                    }

                    HStack {
                        TextField("Type a message…", text: $vm.input)
                            .textFieldStyle(.roundedBorder)
                        Button("Send") {
                            vm.send()
                        }
                        .disabled(vm.input.trimmingCharacters(in: .whitespaces).isEmpty)
                    }
                    .padding()
                }
            }
            .navigationTitle("BLUE COMPUTER")
        }
    }
}

@main
struct MLX_templateApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
