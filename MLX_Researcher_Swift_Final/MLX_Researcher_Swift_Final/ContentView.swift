import SwiftUI
import Combine
import UniformTypeIdentifiers

extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3:
            (a, r, g, b) = (255,
                            (int >> 8) * 17,
                            (int >> 4 & 0xF) * 17,
                            (int & 0xF) * 17)
        case 6:
            (a, r, g, b) = (255,
                            int >> 16,
                            int >> 8 & 0xFF,
                            int & 0xFF)
        case 8:
            (a, r, g, b) = (int >> 24,
                            int >> 16 & 0xFF,
                            int >> 8 & 0xFF,
                            int & 0xFF)
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

struct ChatUISession: Identifiable {
    let id: UUID
    var model: String
    var messages: [String]
    let created: Date
}

struct ContentView: View {
    @StateObject private var vm = ChatViewModel()

    // This is just the UI’s notion of “which model name is selected”
    @State private var selectedModel: String = "BLUECOMPUTER"

    @State private var ChatUISessions: [ChatUISession] = []
    @State private var selectedSessionID: UUID? = nil
    @State private var historyFilterModel: String = "BLUECOMPUTER"

    @State private var thinkingStartDate: Date? = nil
    @State private var thinkingElapsed: Int = 0

    // Model creation UI state (for drag & drop + name)
    @State private var showPDFImporter = false
    @State private var isDropTargeted: Bool = false
    @State private var pendingPDFURL: URL? = nil
    @State private var newModelName: String = ""
    @State private var nameFieldEnabled: Bool = false

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
        Color(hex: "#1266E2"),
        Color(hex: "#663887"),
        Color(hex: "#DE0058")
    ]

    // Picker binding: syncs with current session + calls vm.setModelByName
    private var boundModel: Binding<String> {
        Binding(
            get: {
                if let sessionID = selectedSessionID,
                   let session = ChatUISessions.first(where: { $0.id == sessionID }) {
                    return session.model
                }
                return selectedModel
            },
            set: { newValue in
                selectedModel = newValue
                vm.setModelByName(newValue)

                if let sessionID = selectedSessionID,
                   let index = ChatUISessions.firstIndex(where: { $0.id == sessionID }) {
                    ChatUISessions[index].model = newValue
                }
            }
        )
    }

    private var modelSections: [(key: String, value: [ChatUISession])] {
        Dictionary(grouping: ChatUISessions, by: { $0.model })
            .sorted { $0.key < $1.key }
    }

    var body: some View {
        TabView {
            NavigationStack {
                homeView
            }
            .tabItem {
                Label("Home", systemImage: "house.fill")
            }

            NavigationStack {
                modelPickerView
                    .navigationTitle("Model Selection")
            }
            .tabItem {
                Label("Select Model", systemImage: "cpu")
            }

            historyView
                .tabItem {
                    Label("History", systemImage: "clock")
                }

            NavigationStack {
                settingsView
            }
            .tabItem {
                Label("Settings", systemImage: "gearshape")
            }
        }
        .tabViewStyle(.sidebarAdaptable)
        .onAppear {
            if ChatUISessions.isEmpty {
                let newSession = ChatUISession(
                    id: UUID(),
                    model: selectedModel,
                    messages: [],
                    created: Date()
                )
                ChatUISessions.insert(newSession, at: 0)
                selectedSessionID = newSession.id
                vm.messages = []
                vm.input = ""
            }
            // Make sure VM is on the same starting model
            vm.setModelByName(selectedModel)
        }
        .onChange(of: selectedSessionID) { newValue in
            guard let sessionID = newValue,
                  let session = ChatUISessions.first(where: { $0.id == sessionID }) else {
                vm.messages = []
                vm.input = ""
                return
            }
            vm.messages = session.messages
            selectedModel = session.model
            vm.setModelByName(session.model)
        }
        .onChange(of: vm.isReady) { newValue in
            if !newValue {
                thinkingStartDate = Date()
            } else {
                thinkingStartDate = nil
                thinkingElapsed = 0
            }
        }
    }

    // MARK: - Model Picker + Create New Model

    private var modelPickerView: some View {
        VStack {
            Spacer()

            HStack(alignment: .top, spacing: 32) {

                // LEFT: Model chooser
                VStack(spacing: 16) {
                    Image("Logo")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 80, height: 80)
                        .shadow(radius: 8)

                    Text("Select Model")
                        .font(.title2.bold())

                    Text("Choose which course model to chat with.")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)

                    Picker("Model", selection: boundModel) {
                        ForEach(vm.models, id: \.displayName) { cfg in
                            Text(cfg.displayName).tag(cfg.displayName)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(maxWidth: 220)
                }
                .frame(maxWidth: .infinity, alignment: .top)

                // RIGHT: Create New Model (drag + name + save)
                VStack(alignment: .leading, spacing: 16) {
                    Text("Create New Model")
                        .font(.title3.bold())

                    GeometryReader { geo in
                        let side = max(
                            160.0,
                            min(min(geo.size.width, geo.size.height) * 0.5, 260.0)
                        )

                        ZStack {
                            RoundedRectangle(cornerRadius: 24, style: .continuous)
                                .fill(isDropTargeted ? Color.blue.opacity(0.2) : Color.gray.opacity(0.15))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 24)
                                        .strokeBorder(Color.gray.opacity(0.3), lineWidth: 1)
                                )
                                .frame(width: side, height: side)
                                .overlay {
                                    // 🔹 NEW: show filename if a PDF is selected, otherwise default text
                                    VStack(spacing: 8) {
                                        if let url = pendingPDFURL {
                                            Image(systemName: "doc.fill")
                                                .font(.system(size: 40, weight: .regular))
                                                .foregroundColor(.secondary)

                                            Text(url.lastPathComponent)
                                                .font(.headline)
                                                .foregroundColor(.secondary)
                                                .multilineTextAlignment(.center)
                                                .lineLimit(2)
                                                .truncationMode(.middle)

                                            Text("Click to change file")
                                                .font(.caption)
                                                .foregroundColor(.secondary)
                                        } else {
                                            Image(systemName: "arrow.up.doc")
                                                .font(.system(size: 40, weight: .regular))
                                                .foregroundColor(.secondary)

                                            Text("Drop PDF here")
                                                .font(.headline)
                                                .foregroundColor(.secondary)

                                            Text("or click to choose")
                                                .font(.caption)
                                                .foregroundColor(.secondary)
                                        }
                                    }
                                    .padding()
                                }
                                .contentShape(Rectangle())
                                .onTapGesture {
                                    showPDFImporter = true
                                }
                                .onDrop(of: [UTType.fileURL], isTargeted: $isDropTargeted) { providers in
                                    guard let provider = providers.first else { return false }

                                    provider.loadItem(
                                        forTypeIdentifier: UTType.fileURL.identifier,
                                        options: nil
                                    ) { item, error in
                                        if let error = error {
                                            print("Drop error: \(error.localizedDescription)")
                                            return
                                        }

                                        if let data = item as? Data,
                                           let url = URL(dataRepresentation: data, relativeTo: nil) {
                                            handleNewPDF(url: url)
                                        } else if let url = item as? URL {
                                            handleNewPDF(url: url)
                                        } else {
                                            print("Drop: unsupported item \(String(describing: item))")
                                        }
                                    }

                                    return true
                                }
                        }
                    }
                    .frame(height: 220)

                    VStack(alignment: .leading, spacing: 6) {
                        Text("Model Name")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        TextField("Model Name", text: $newModelName)
                            .disabled(!nameFieldEnabled)
                            .textFieldStyle(.roundedBorder)
                            .opacity(nameFieldEnabled ? 1.0 : 0.4)
                    }

                    Button("Save Model") {
                        guard let pdfURL = pendingPDFURL else { return }
                        let trimmed = newModelName.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !trimmed.isEmpty else { return }

                        // Create model in the VM
                        vm.createModel(displayName: trimmed, ragPDFURL: pdfURL)
                        vm.setModelByName(trimmed)

                        // Update current session to use this new model
                        selectedModel = trimmed
                        if let sessionID = selectedSessionID,
                           let index = ChatUISessions.firstIndex(where: { $0.id == sessionID }) {
                            ChatUISessions[index].model = trimmed
                        }

                        // Reset UI state – this also resets the drop area label
                        pendingPDFURL = nil
                        newModelName = ""
                        nameFieldEnabled = false
                    }
                    .buttonStyle(.borderedProminent)
                    .clipShape(RoundedRectangle(cornerRadius: 30))
                    .disabled(pendingPDFURL == nil || newModelName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
                .frame(maxWidth: .infinity, alignment: .top)
            }
            .padding(.horizontal, 32)

            Spacer()
        }
        .padding(.vertical, 24)
        .fileImporter(
            isPresented: $showPDFImporter,
            allowedContentTypes: [.pdf],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first {
                    handleNewPDF(url: url)
                }
            case .failure(let error):
                print("fileImporter error: \(error.localizedDescription)")
            }
        }
    }

    private func handleNewPDF(url: URL) {
        // Only accept PDFs
        guard url.pathExtension.lowercased() == "pdf" else {
            print("Selected file is not a PDF: \(url.lastPathComponent)")
            return
        }
        Task { @MainActor in
            pendingPDFURL = url       // <- drives the filename label
            nameFieldEnabled = true
            print("New PDF ready for model creation: \(url.path)")
        }
    }

    // MARK: - Model/Embedder loading overlays

    private var modelLoadingOverlay: some View {
        Group {
            if vm.isModelLoading {
                ZStack {
                    Color.black.opacity(0.6)
                        .ignoresSafeArea()

                    VStack(spacing: 20) {
                        Image("Logo")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 80, height: 80)
                            .shadow(radius: 10)

                        if let progress = vm.modelLoadProgress {
                            VStack(spacing: 12) {
                                Text("Loading Model...")
                                    .font(.title3.bold())

                                ProgressView(value: progress.fractionCompleted) {
                                    Text("\(Int(progress.fractionCompleted * 100))%")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                .progressViewStyle(.linear)
                                .frame(width: 200)
                                .tint(.blue)
                            }
                        } else {
                            VStack(spacing: 12) {
                                Text("Loading Model...")
                                    .font(.title3.bold())

                                ProgressView()
                                    .progressViewStyle(.circular)
                                    .scaleEffect(1.2)
                            }
                        }

                        Text("Please wait...")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .padding(40)
                    .background(
                        RoundedRectangle(cornerRadius: 20)
                            .fill(.regularMaterial)
                            .shadow(color: .black.opacity(0.3),
                                    radius: 20)
                    )
                }
                .transition(.opacity)
                .animation(.easeInOut, value: vm.isModelLoading)
            }
        }
    }

    private var embeddermodelLoadingOverlay: some View {
        Group {
            if vm.isEmbedModelLoading {
                ZStack {
                    Color.black.opacity(0.6)
                        .ignoresSafeArea()

                    VStack(spacing: 20) {
                        Image(systemName: "doc.text.magnifyingglass")
                            .font(.system(size: 60))
                            .foregroundStyle(.blue.gradient)

                        if let progress = vm.embedModelProgress {
                            VStack(spacing: 12) {
                                Text("Loading Embedder...")
                                    .font(.title3.bold())

                                ProgressView(value: progress.fractionCompleted) {
                                    Text("\(Int(progress.fractionCompleted * 100))%")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                .progressViewStyle(.linear)
                                .frame(width: 200)
                                .tint(.blue)
                            }
                        } else {
                            VStack(spacing: 12) {
                                Text("Loading Embedder...")
                                    .font(.title3.bold())

                                ProgressView()
                                    .progressViewStyle(.circular)
                                    .scaleEffect(1.2)
                            }
                        }

                        Text("Preparing embeddings...")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .padding(40)
                    .background(
                        RoundedRectangle(cornerRadius: 20)
                            .fill(.regularMaterial)
                            .shadow(color: .black.opacity(0.3),
                                    radius: 20)
                    )
                }
                .transition(.opacity)
                .animation(.easeInOut, value: vm.isEmbedModelLoading)
            }
        }
    }

    // MARK: - Home View (chat UI)

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
                    let newSession = ChatUISession(
                        id: UUID(),
                        model: selectedModel,
                        messages: [],
                        created: Date()
                    )
                    ChatUISessions.insert(newSession, at: 0)
                    selectedSessionID = newSession.id
                    vm.messages = []
                    vm.input = ""
                }) {
                    Label("New Chat", systemImage: "plus")
                }
                .buttonStyle(.borderedProminent)
                .clipShape(RoundedRectangle(cornerRadius: 30))
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
        .navigationTitle("AVELA-CourseSLM")
        .onChange(of: vm.messages) { newMessages in
            guard let sessionID = selectedSessionID,
                  let index = ChatUISessions.firstIndex(where: { $0.id == sessionID }) else {
                return
            }
            ChatUISessions[index].messages = newMessages
        }
        .overlay(modelLoadingOverlay)
        .overlay(embeddermodelLoadingOverlay)
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
                            .buttonStyle(.plain)
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
                            .buttonStyle(.plain)
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
                        MessageBubble(message: message,
                                      colorIndex: index,
                                      chatColors: chatColors)
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
                HStack(alignment: .top, spacing: 16) {
                    TextField("Type a message...",
                              text: $vm.input,
                              axis: .vertical)
                        .textFieldStyle(.plain)
                        .font(.system(size: 16))
                        .padding(.horizontal, 24)
                        .padding(.vertical, 24)
                        .frame(minHeight: 120, alignment: .topLeading)
                        .background(
                            RoundedRectangle(cornerRadius: 28)
                                .fill(Color.gray.opacity(0.1))
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 28)
                                .stroke(Color.gray.opacity(0.3), lineWidth: 1.5)
                        )
                        .lineLimit(1...12)
                        .disabled(!vm.isReady)

                    Button("Send") {
                        vm.send()
                    }
                    .buttonStyle(.borderedProminent)
                    .clipShape(RoundedRectangle(cornerRadius: 30))
                    .padding(.top, 20)
                    .disabled(vm.input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !vm.isReady)
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 24)

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
        .onReceive(
            Timer.publish(every: 1, on: .main, in: .common).autoconnect()
        ) { _ in
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
                        let newSession = ChatUISession(
                            id: UUID(),
                            model: selectedModel,
                            messages: [],
                            created: Date()
                        )
                        ChatUISessions.insert(newSession, at: 0)
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
                        ForEach(vm.models, id: \.displayName) { cfg in
                            Text(cfg.displayName).tag(cfg.displayName)
                        }
                    }
                    .pickerStyle(.segmented)
                }
                .padding([.top, .horizontal])

                Section(header: Text("\(historyFilterModel)")) {
                    ForEach(ChatUISessions.filter { $0.model == historyFilterModel }.prefix(5)) { session in
                        VStack(alignment: .leading, spacing: 4) {
                            Button {
                                selectedSessionID = session.id
                                vm.messages = session.messages
                                selectedModel = session.model
                                vm.setModelByName(session.model)
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
            }
            .navigationTitle("History")
        }
    }

    // MARK: - Settings View

    private var settingsView: some View {
    #if os(macOS)
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
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
                }
                Divider()

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

// Message bubble
struct MessageBubble: View {
    let message: String
    let colorIndex: Int
    let chatColors: [Color]

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
        HStack(alignment: .top, spacing: 12) {
            if isUser {
                Spacer(minLength: 60)
                Text(displayText)
                    .font(.body)
                    .foregroundColor(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: 18)
                            .fill(chatColors[colorIndex % chatColors.count])
                    )
                Image(systemName: "person.fill")
                    .foregroundColor(.white)
                    .background(
                        Circle()
                            .fill(Color.gray)
                            .frame(width: 36, height: 36)
                    )
                    .frame(width: 36, height: 36)
            } else {
                Image("Logo")
                    .resizable()
                    .scaledToFill()
                    .frame(width: 36, height: 36)
                    .clipShape(Circle())
                    .background(
                        Circle()
                            .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                    )
                Text(displayText)
                    .font(.body)
                    .foregroundColor(.primary)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: 18)
                            .fill(Color.gray.opacity(0.2))
                    )
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
