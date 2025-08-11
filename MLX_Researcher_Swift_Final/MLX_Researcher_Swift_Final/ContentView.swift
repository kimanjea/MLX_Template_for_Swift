import SwiftUI
import Combine

extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3: // RGB (12-bit)
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: // RGB (24-bit)
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8: // ARGB (32-bit)
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (255, 0, 0, 0)
        }
        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue: Double(b) / 255,
            opacity: Double(a) / 255
        )
    }
}

// 1. Add a struct for a chat session at the top (for demo purposes).
struct ChatSession: Identifiable {
    let id: UUID
    var model: String
    var messages: [String]
    let created: Date
}

struct ContentView: View {
    @StateObject private var vm = ChatViewModel()
    @State private var selectedModel: String = "Gemma"
    // 2. Replace single session ID with multi-session state:
    @State private var chatSessions: [ChatSession] = []
    @State private var selectedSessionID: UUID? = nil
    @State private var historyFilterModel: String = "Gemma"
    
    // Added state for thinking timer
    @State private var thinkingStartDate: Date? = nil
    @State private var thinkingElapsed: Int = 0
    
    private let suggestedQuestions = [
        "What is data activism?",
        "What is a variable?",
        "What is Python?",
        "What is a function?",
        "Examples of Data Activism"
    ]
    
    private let welcomeColors: [Color] = [
        Color(hex: "#DE0058"),
        Color(hex: "#00B500"),
        Color(hex: "#EDC300"),
        Color(hex: "#1266E2"),
        Color(hex: "#663887")
    ]
    
    private let chatColors: [Color] = [
        Color(hex: "#DE0058"),
        Color(hex: "#00B500"),
        Color(hex: "#EDC300"),
        Color(hex: "#1266E2"),
        Color(hex: "#663887")
    ]
    
    private var boundModel: Binding<String> {
        Binding(
            get: {
                if let sessionID = selectedSessionID,
                   let session = chatSessions.first(where: { $0.id == sessionID }) {
                    return session.model
                }
                return selectedModel
            },
            set: { newValue in
                selectedModel = newValue
                if let sessionID = selectedSessionID,
                   let index = chatSessions.firstIndex(where: { $0.id == sessionID }) {
                    chatSessions[index].model = newValue
                }
            }
        )
    }
    
    private var modelSections: [(key: String, value: [ChatSession])] {
        Dictionary(grouping: chatSessions, by: { $0.model })
            .sorted { $0.key < $1.key }
    }
    
    var body: some View {
        TabView {
            Tab("Home", systemImage: "house.fill") {
                homeView
            }
            
            TabSection("History") {
                Tab("Recent", systemImage: "clock") {
                    historyView
                }
            }
            .defaultVisibility(.hidden, for: .tabBar)
            
            TabSection("Settings") {
                Tab("Settings", systemImage: "gearshape") {
                    settingsView
                }
            }
            .defaultVisibility(.hidden, for: .tabBar)
        }
        .tabViewStyle(.sidebarAdaptable)
        .onAppear {
            if chatSessions.isEmpty {
                let newSession = ChatSession(id: UUID(), model: selectedModel, messages: [], created: Date())
                chatSessions.insert(newSession, at: 0)
                selectedSessionID = newSession.id
                vm.messages = []
                vm.input = ""
            }
        }
        .onChange(of: selectedSessionID) { newValue in
            guard let sessionID = newValue,
                  let session = chatSessions.first(where: { $0.id == sessionID }) else {
                vm.messages = []
                vm.input = ""
                return
            }
            vm.messages = session.messages
            selectedModel = session.model
        }
        .onChange(of: vm.isReady) { newValue in
            if newValue == false {
                thinkingStartDate = Date()
            } else {
                thinkingStartDate = nil
                thinkingElapsed = 0
            }
        }
    }
    
    // MARK: - Home View (Chat Interface)
    private var homeView: some View {
        VStack(spacing: 0) {
            Image("Logo")
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 84, height: 84)
                .shadow(radius: 8)
                .padding(.top, 24)
            
            HStack(spacing: 16) {
                Button(action: {
                    // 3. Append new session and select it
                    let newSession = ChatSession(id: UUID(), model: selectedModel, messages: [], created: Date())
                    chatSessions.insert(newSession, at: 0)
                    selectedSessionID = newSession.id
                    vm.messages = []
                    vm.input = ""
                }) {
                    Label("New Chat", systemImage: "plus")
                }
                .buttonStyle(.borderedProminent)
                .clipShape(RoundedRectangle(cornerRadius: 30))
                
                Picker("Model", selection: boundModel) {
                    Text("Gemma").tag("Gemma")
                    Text("BLUECOMPUTER.2").tag("BLUECOMPUTER.2")
                    Text("ChatGPT-4o-Mini").tag("ChatGPT-4o-Mini")
                }
                .pickerStyle(.menu)
                .frame(maxWidth: 180)
            }
            .padding([.top, .horizontal])
            
            if vm.messages.isEmpty {
                welcomeView
            } else {
                messagesView
            }
            inputView
        }
        .id(selectedSessionID ?? UUID())
        .navigationTitle("Chat")
        .onChange(of: vm.messages) { newMessages in
            // 4. Update current session's messages when vm.messages changes (e.g. after sending)
            guard let sessionID = selectedSessionID,
                  let index = chatSessions.firstIndex(where: { $0.id == sessionID }) else {
                return
            }
            chatSessions[index].messages = newMessages
        }
        
    }
    
    private var welcomeView: some View {
        VStack(spacing: 32) {
            Spacer()
            VStack(spacing: 16) {
                Text("Welcome to AVELA AI")
                    .font(.largeTitle.bold())
                Text("Click to learn about data activism.")
                    .font(.title3)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                
                VStack(spacing: 12) {
                    HStack(spacing: 12) {
                        ForEach(Array(suggestedQuestions.prefix(3).enumerated()), id: \.element) { (index, question) in
                            Button(action: {
                                vm.input = question
                                vm.send()
                            }) {
                                Text(question)
                                    .font(.callout)
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 18)
                                    .padding(.vertical, 14)
                                    .background(
                                        RoundedRectangle(cornerRadius: 30)
                                            .fill(welcomeColors[index % welcomeColors.count])
                                    )
                            }
                            .buttonStyle(.plain) // ✅ Removes extra rectangle
                            .accessibilityLabel(question)
                        }
                    }
                    HStack(spacing: 12) {
                        ForEach(Array(suggestedQuestions.suffix(2).enumerated()), id: \.element) { (index, question) in
                            Button(action: {
                                vm.input = question
                                vm.send()
                            }) {
                                Text(question)
                                    .font(.callout)
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 18)
                                    .padding(.vertical, 14)
                                    .background(
                                        RoundedRectangle(cornerRadius: 30)
                                            .fill(welcomeColors[(index + 3) % welcomeColors.count])
                                    )
                            }
                            .buttonStyle(.plain) // ✅ Removes extra rectangle
                            .accessibilityLabel(question)
                        }
                    }
                }
                .padding(.vertical)
            }
            Spacer()
        }
        .padding()
    }

    
    private var messagesView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(Array(vm.messages.enumerated()), id: \.offset) { index, message in
                        MessageBubble(message: message, color: chatColors[index % chatColors.count])
                            .id(index)
                    }
                }
                .padding()
            }
            .onChange(of: vm.messages.count) { _ in
                if let lastIndex = vm.messages.indices.last {
                    withAnimation(.easeInOut(duration: 0.5)) {
                        proxy.scrollTo(lastIndex, anchor: .bottom)
                    }
                }
            }
        }
    }
    
    private var inputView: some View {
        ZStack {
            VStack(spacing: 0) {
                Divider()
                HStack(spacing: 12) {
                    TextField("Type a message...", text: $vm.input, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .lineLimit(1...4)
                        .disabled(!vm.isReady)
                    Button("Send") {
                        vm.send()
                    }
                    .buttonStyle(.borderedProminent)
                    .clipShape(RoundedRectangle(cornerRadius: 30))
                    .disabled(vm.input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !vm.isReady)
                }
                .padding()
                
                if !vm.isReady {
                    Text("Thinking for \(thinkingElapsed) second(s)...")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .padding(.horizontal)
                        .padding(.bottom, 6)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
#if os(macOS)
            .background(Color(NSColor.controlBackgroundColor))
#else
            .background(Color(.systemBackground))
#endif
        }
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .padding(.horizontal, 8)
        .onReceive(Timer.publish(every: 1, on: .main, in: .common).autoconnect()) { _ in
            if let start = thinkingStartDate, !vm.isReady {
                thinkingElapsed = Int(Date().timeIntervalSince(start))
            } else {
                thinkingElapsed = 0
            }
        }
    }
    
    // MARK: - History View
    private var historyView: some View {
        NavigationStack {
            List {
                VStack(alignment: .leading, spacing: 12) {
                    Button(action: {
                        // 3. Append new session and select it
                        let newSession = ChatSession(id: UUID(), model: selectedModel, messages: [], created: Date())
                        chatSessions.insert(newSession, at: 0)
                        selectedSessionID = newSession.id
                        vm.messages = []
                        vm.input = ""
                    }) {
                        Label("New Chat", systemImage: "plus")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .clipShape(RoundedRectangle(cornerRadius: 30))
                    
                    Picker("Model", selection: $historyFilterModel) {
                        Text("Gemma").tag("Gemma")
                        Text("BLUECOMPUTER.2").tag("BLUECOMPUTER.2")
                        Text("ChatGPT-4o-Mini").tag("ChatGPT-4o-Mini")
                    }
                    .pickerStyle(.segmented)
                }
                .padding([.top, .horizontal])
                
                // Show only conversations matching the selected model
                Section(header: Text("\(historyFilterModel)")) {
                    ForEach(chatSessions.filter { $0.model == historyFilterModel }.prefix(5)) { session in
                        VStack(alignment: .leading, spacing: 4) {
                            Button {
                                selectedSessionID = session.id
                                vm.messages = session.messages
                                selectedModel = session.model
                            } label: {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Conversation \(session.id.uuidString.prefix(5))")
                                        .font(.headline)
                                    if let lastMessage = session.messages.last {
                                        Text(lastMessage)
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                            .lineLimit(1)
                                    } else {
                                        Text("No messages yet")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                    Text(session.created, style: .relative)
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                }
                            }
                            .buttonStyle(.plain)
                        }
                        .padding(.vertical, 2)
                    }
                }
                /*
                // 7. Remove older conversations for now
                Section("Older Conversations") {
                    ForEach(5..<10, id: \.self) { index in
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Conversation \(index + 1)")
                                .font(.headline)
                            Text("Python and data analysis...")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("\(index - 4) days ago")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                        .padding(.vertical, 2)
                    }
                }
                */
            }
            .navigationTitle("History")
        }
    }
    
    // MARK: - Settings View
    private var settingsView: some View {
    #if os(macOS)
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                // Appearance Section
                Text("Appearance").font(.title3.bold()).padding(.bottom, 6)
                HStack {
                    Label("Theme", systemImage: "paintbrush")
                    Spacer()
                    Text("System").foregroundColor(.secondary)
                }
                HStack {
                    Label("Text Size", systemImage: "textformat.size")
                    Spacer()
                    Text("Medium").foregroundColor(.secondary)
                }
                Divider()

                // Behavior Section
                Text("Behavior").font(.title3.bold()).padding(.bottom, 6)
                HStack {
                    Label("Auto-send on Return", systemImage: "return")
                    Spacer()
                    Toggle("", isOn: .constant(false))
                }
                HStack {
                    Label("Save History", systemImage: "externaldrive")
                    Spacer()
                    Toggle("", isOn: .constant(true))
                }
                HStack {
                    Label("Smart Suggestions", systemImage: "lightbulb")
                    Spacer()
                    Toggle("", isOn: .constant(true))
                }
                Divider()

                // Privacy Section
                Text("Privacy").font(.title3.bold()).padding(.bottom, 6)
                HStack {
                    Label("Analytics", systemImage: "chart.bar")
                    Spacer()
                    Toggle("", isOn: .constant(false))
                }
                Button(role: .destructive) {
                    // Clear history action
                } label: {
                    Label("Clear All History", systemImage: "trash")
                        .clipShape(RoundedRectangle(cornerRadius: 30))
                }
                Divider()

                // About Section
                Text("About").font(.title3.bold()).padding(.bottom, 6)
                HStack {
                    Label("Version", systemImage: "info.circle")
                    Spacer()
                    Text("2.0.1").foregroundColor(.secondary)
                }
                HStack {
                    Label("Build", systemImage: "hammer")
                    Spacer()
                    Text("2024.08.08").foregroundColor(.secondary)
                }
                Button {
                    // Show licenses
                } label: {
                    Label("Open Source Licenses", systemImage: "doc.text")
                }

            }
            .padding(32)
            .frame(maxWidth: 500)
        }
        .navigationTitle("Settings")
    #else
        // iPad/iOS
        Form {
            Section("Appearance") {
                HStack {
                    Label("Theme", systemImage: "paintbrush")
                    Spacer()
                    Text("System").foregroundColor(.secondary)
                }
                HStack {
                    Label("Text Size", systemImage: "textformat.size")
                    Spacer()
                    Text("Medium").foregroundColor(.secondary)
                }
            }
            Section("Behavior") {
                HStack {
                    Label("Auto-send on Return", systemImage: "return")
                    Spacer()
                    Toggle("", isOn: .constant(false))
                }
                HStack {
                    Label("Save History", systemImage: "externaldrive")
                    Spacer()
                    Toggle("", isOn: .constant(true))
                }
                HStack {
                    Label("Smart Suggestions", systemImage: "lightbulb")
                    Spacer()
                    Toggle("", isOn: .constant(true))
                }
            }
            Section("Privacy") {
                HStack {
                    Label("Analytics", systemImage: "chart.bar")
                    Spacer()
                    Toggle("", isOn: .constant(false))
                }
                Button {
                    // Clear history action
                } label: {
                    Label("Clear All History", systemImage: "trash")
                        .foregroundColor(.red)
                        .clipShape(RoundedRectangle(cornerRadius: 30))
                }
            }
            Section("About") {
                HStack {
                    Label("Version", systemImage: "info.circle")
                    Spacer()
                    Text("2.0.1").foregroundColor(.secondary)
                }
                HStack {
                    Label("Build", systemImage: "hammer")
                    Spacer()
                    Text("2024.08.08").foregroundColor(.secondary)
                }
                Button {
                    // Show licenses
                } label: {
                    Label("Open Source Licenses", systemImage: "doc.text")
                }
            }
        }
        .navigationTitle("Settings")
    #endif
    }
}

// Message bubble component
struct MessageBubble: View {
    let message: String
    let color: Color
    
    private var isUser: Bool {
        message.starts(with: "You:")
    }
    
    private var displayText: String {
        if isUser {
            return String(message.dropFirst(4))
        } else if message.starts(with: "Bot:") {
            return String(message.dropFirst(4))
        }
        return message
    }
    
    var body: some View {
        HStack {
            if isUser {
                Spacer(minLength: 60)
            }
            
            Text(displayText)
                .font(.body)
                .foregroundColor(.white)
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(
                    RoundedRectangle(cornerRadius: 18)
                        .fill(color)
                )
            
            if !isUser {
                Spacer(minLength: 60)
            }
        }
    }
}

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        var chunks: [[Element]] = []
        var index = 0
        while index < self.count {
            let end = Swift.min(index + size, self.count)
            chunks.append(Array(self[index..<end]))
            index += size
        }
        return chunks
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
