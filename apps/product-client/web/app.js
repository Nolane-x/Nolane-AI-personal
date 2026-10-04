(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const els = {
    conversation: $("conversation"),
    emptyState: $("emptyState"),
    emptyTitle: $("emptyTitle"),
    emptyBody: $("emptyBody"),
    powerButton: $("powerButton"),
    powerButtonLabel: $("powerButtonLabel"),
    powerStatusText: $("powerStatusText"),
    composer: $("composer"),
    input: $("messageInput"),
    send: $("sendButton"),
    banner: $("runtimeBanner"),
    bannerText: $("runtimeBannerText"),
    retry: $("runtimeRetry"),
    profileButton: $("profileButton"),
    dialog: $("personalizationDialog"),
    closeProfile: $("closeProfileButton"),
    profileForm: $("profileForm"),
    saveState: $("profileSaveState"),
    preferredName: $("preferredName"),
    language: $("language"),
    responseLength: $("responseLength"),
    conversationStyle: $("conversationStyle"),
    initiative: $("initiative"),
    memoryEnabled: $("memoryEnabled"),
    personalInstruction: $("personalInstruction"),
    connectionDetails: $("connectionDetails"),
    remoteEndpoint: $("remoteEndpoint"),
    remoteToken: $("remoteToken"),
    saveConnection: $("saveConnection"),
    clearConnection: $("clearConnection"),
    identitySubline: $("identitySubline"),
    learningDetails: $("learningDetails"),
    learningSummary: $("learningSummary"),
    prepareLearningWindow: $("prepareLearningWindow"),
    openLearningReview: $("openLearningReview"),
    learningDialog: $("learningDialog"),
    learningBody: document.querySelector(".learning-body"),
    closeLearning: $("closeLearningButton"),
    learningProgress: $("learningProgress"),
    reviewCandidate: $("reviewCandidate"),
    reviewPrompt: $("reviewPrompt"),
    reviewTarget: $("reviewTarget"),
    reviewLanguage: $("reviewLanguage"),
    reviewComplete: $("reviewComplete"),
    reviewActions: $("reviewActions"),
    approveLearning: $("approveLearning"),
    rejectLearning: $("rejectLearning"),
    markSensitive: $("markSensitive"),
    finalizeLearning: $("finalizeLearning"),
    uiLanguage: $("uiLanguage"),
    mindStateButton: $("mindStateButton"),
    mindStateLabel: $("mindStateLabel"),
    mindOrb: $("mindOrb"),
    mindDialog: $("mindDialog"),
    closeMind: $("closeMindButton"),
    mindHeroOrb: $("mindHeroOrb"),
    mindMood: $("mindMood"),
    emotionChips: $("emotionChips"),
    mindActivity: $("mindActivity"),
    mindProgressBar: $("mindProgressBar"),
    mindIntent: $("mindIntent"),
    mindConfidence: $("mindConfidence"),
    mindThreads: $("mindThreads"),
    closenessBar: $("closenessBar"),
    trustBar: $("trustBar"),
    familiarityBar: $("familiarityBar"),
    assistantAvatarButton: $("assistantAvatarButton"),
    assistantAvatarInput: $("assistantAvatarInput"),
    assistantAvatarFallback: $("assistantAvatarFallback"),
    assistantAvatarImage: $("assistantAvatarImage"),
    assistantNameButton: $("assistantNameButton"),
    assistantNameInput: $("assistantNameInput"),
    appShell: $("appShell"),
    mindMemories: $("mindMemories"),
    mindRelationshipStage: $("mindRelationshipStage"),
    proactiveWrap: $("proactiveWrap"),
    proactiveCapsule: $("proactiveCapsule"),
    proactiveTitle: $("proactiveTitle"),
    proactiveBody: $("proactiveBody"),
    proactiveDismiss: $("proactiveDismiss"),
    onboardingDialog: $("onboardingDialog"),
    onboardingForm: $("onboardingForm"),
    onboardingName: $("onboardingName"),
    onboardingLocale: $("onboardingLocale"),
    onboardingStyle: $("onboardingStyle"),
  };

  const strings = {
    vi: {
      off: "Đang tắt",
      starting: "Đang khởi động",
      on: "Đang chạy",
      thinking: "Đang nghĩ",
      error: "Cần chú ý",
      turnOn: "Bật AI",
      turnOff: "Tắt AI",
      emptyOffTitle: "Nolane đang nghỉ.",
      emptyOffBody: "Bật AI khi bạn muốn nói chuyện.",
      emptyOnTitle: "Mình ở đây.",
      emptyOnBody: "Cứ nói điều bạn đang nghĩ.",
      input: "Nhắn cho Nolane…",
      send: "Gửi",
      retry: "Thử lại",
      unavailable: "Runtime chưa sẵn sàng.",
      modelMissing: "Thiếu model hoặc tokenizer của bản phát hành.",
      remoteMissing: "Android chưa được ghép với runtime Nolane.",
      saved: "Đã lưu",
      saveError: "Không thể lưu",
      connectionSaved: "Đã lưu kết nối",
      connectionCleared: "Đã xóa kết nối",
      settingsSubline: "của riêng bạn",
      learningNone: "Chưa có lượt duyệt.",
      learningUnavailable: "Chưa khả dụng trên runtime này.",
      learningReady: "lượt đã sẵn sàng",
      learningReviewed: "đã duyệt",
      learningPreparing: "Đang chuẩn bị lượt duyệt…",
      learningNeedMore: "Chưa đủ đoạn chat mới để tạo một lượt duyệt sạch.",
      learningDone: "Đã duyệt hết lượt này.",
    },
    en: {
      off: "Off",
      starting: "Starting",
      on: "Running",
      thinking: "Thinking",
      error: "Needs attention",
      turnOn: "Start AI",
      turnOff: "Stop AI",
      emptyOffTitle: "Nolane is resting.",
      emptyOffBody: "Start the AI whenever you want to talk.",
      emptyOnTitle: "I'm here.",
      emptyOnBody: "Say whatever is on your mind.",
      input: "Message Nolane…",
      send: "Send",
      retry: "Try again",
      unavailable: "The runtime is not ready.",
      modelMissing: "Release model or tokenizer assets are missing.",
      remoteMissing: "Android is not paired with a Nolane runtime yet.",
      saved: "Saved",
      saveError: "Could not save",
      connectionSaved: "Connection saved",
      connectionCleared: "Connection cleared",
      settingsSubline: "yours",
      learningNone: "No review window yet.",
      learningUnavailable: "Not available on this runtime.",
      learningReady: "windows ready",
      learningReviewed: "reviewed",
      learningPreparing: "Preparing a review window…",
      learningNeedMore: "Not enough new chat yet for a clean review window.",
      learningDone: "This review window is complete.",
    },
  };

  Object.assign(strings.en, {
    profileAria: "Personalize",
    conversationAria: "Conversation with Nolane",
    mindOpen: "Open Nolane mind state",
    settingsEyebrow: "Personalization",
    settingsTitle: "Make Nolane feel more like yours",
    preferredNameLabel: "What Nolane calls you",
    preferredNamePlaceholder: "Your preferred name",
    responseLanguageLabel: "AI response language",
    lengthLabel: "Response length",
    styleLabel: "Voice",
    initiativeLabel: "Initiative",
    memoryTitle: "Remember you",
    memoryBody: "Use local memories to understand you over time.",
    instructionLabel: "One thing Nolane should always know",
    saveProfile: "Save",
    mindEyebrow: "Observable mind",
    mindTitle: "Inside Nolane",
    mindDisclaimer: "A live view of state and activity — not raw private reasoning.",
    moodLabel: "Mood",
    activityLabel: "Activity",
    activityReady: "Ready to listen",
    activityReading: "Taking in your message",
    activityContext: "Bringing context together",
    activityShaping: "Shaping a response",
    activityHint: "A visible activity summary, not a transcript of private reasoning.",
    intentLabel: "Intent",
    intentConversation: "Conversation",
    intentReply: "Respond thoughtfully",
    confidenceLabel: "Confidence",
    confidenceClear: "Clear",
    confidenceSteady: "Steady",
    confidenceUncertain: "Uncertain",
    confidenceHint: "Based on the runtime's current uncertainty signal.",
    threadsLabel: "In mind",
    noThreads: "No open thread right now.",
    relationshipLabel: "Relationship",
    closeness: "Closeness",
    trust: "Trust",
    familiarity: "Familiarity",
    moodCalm: "Calm",
    moodCurious: "Curious",
    moodWarm: "Warm",
    moodConcerned: "Concerned",
    moodPlayful: "Playful",
    moodIrritated: "Irritated",
    moodLow: "Quiet",
    emotionCuriosity: "Curiosity",
    emotionConcern: "Concern",
    emotionPlayful: "Playful",
    emotionEnergy: "Energy",
    emotionIrritation: "Irritation",
    changeAvatar: "Change AI avatar",
    renameAi: "Rename AI",
    aiName: "AI name",
    memoryThreadsLabel: "Memory & threads",
    openThreadsTitle: "Unfinished threads",
    memoriesTitle: "What Nolane remembers",
    noMemories: "No local memory yet.",
    memoryKeep: "Keep",
    memoryEdit: "Edit",
    memorySave: "Save",
    memoryCancel: "Cancel",
    memoryDelete: "Forget",
    memoryDeleteConfirm: "Forget?",
    relationshipStageNew: "New",
    relationshipStageFamiliar: "Familiar",
    relationshipStageClose: "Close",
    proactiveTitle: "Nolane has something to say",
    proactiveThreadBody: "Still thinking about something unfinished.",
    proactiveBody: "Tap to open. Ignore it and Nolane stays quiet.",
    proactiveDismiss: "Dismiss",
    onboardingEyebrow: "First meeting",
    onboardingTitle: "Make Nolane yours in three small steps",
    onboardingBody: "Only the essentials. You can change everything later.",
    onboardingNameLabel: "What should Nolane call you?",
    onboardingLanguageLabel: "Interface language",
    onboardingStyleLabel: "How should Nolane talk?",
    onboardingPrivacy: "Stored locally with your profile.",
    onboardingStart: "Start talking",
  });
  Object.assign(strings.vi, {
    profileAria: "Cá nhân hóa",
    conversationAria: "Cuộc trò chuyện với Nolane",
    mindOpen: "Mở trạng thái nội tâm của Nolane",
    settingsEyebrow: "Cá nhân hóa",
    settingsTitle: "Để Nolane hợp với bạn hơn",
    preferredNameLabel: "Nolane gọi bạn là",
    preferredNamePlaceholder: "Tên bạn thích",
    responseLanguageLabel: "Ngôn ngữ AI trả lời",
    lengthLabel: "Độ dài trả lời",
    styleLabel: "Cách nói",
    initiativeLabel: "Chủ động",
    memoryTitle: "Nhớ về bạn",
    memoryBody: "Dùng ký ức cục bộ để hiểu bạn theo thời gian.",
    instructionLabel: "Một điều Nolane nên luôn biết",
    saveProfile: "Lưu",
    mindEyebrow: "Nội tâm có thể quan sát",
    mindTitle: "Bên trong Nolane",
    mindDisclaimer: "Trạng thái và hoạt động đang diễn ra — không phải chuỗi suy luận riêng tư thô.",
    moodLabel: "Tâm trạng",
    activityLabel: "Hoạt động",
    activityReady: "Sẵn sàng lắng nghe",
    activityReading: "Đang tiếp nhận lời bạn nói",
    activityContext: "Đang ghép ngữ cảnh lại",
    activityShaping: "Đang hình thành phản hồi",
    activityHint: "Đây là tóm tắt hoạt động có thể quan sát, không phải bản chép suy luận riêng tư.",
    intentLabel: "Ý định",
    intentConversation: "Trò chuyện",
    intentReply: "Trả lời cẩn thận",
    confidenceLabel: "Độ chắc chắn",
    confidenceClear: "Rõ",
    confidenceSteady: "Khá chắc",
    confidenceUncertain: "Chưa chắc",
    confidenceHint: "Dựa trên tín hiệu uncertainty hiện tại của runtime.",
    threadsLabel: "Đang để tâm",
    noThreads: "Hiện chưa có chủ đề dang dở.",
    relationshipLabel: "Mối quan hệ",
    closeness: "Gần gũi",
    trust: "Tin tưởng",
    familiarity: "Quen thuộc",
    moodCalm: "Bình tĩnh",
    moodCurious: "Tò mò",
    moodWarm: "Ấm áp",
    moodConcerned: "Hơi lo",
    moodPlayful: "Tinh nghịch",
    moodIrritated: "Khó chịu",
    moodLow: "Trầm",
    emotionCuriosity: "Tò mò",
    emotionConcern: "Quan tâm",
    emotionPlayful: "Tinh nghịch",
    emotionEnergy: "Năng lượng",
    emotionIrritation: "Khó chịu",
    changeAvatar: "Đổi ảnh đại diện AI",
    renameAi: "Đổi tên AI",
    aiName: "Tên AI",
    memoryThreadsLabel: "Ký ức & chuyện dang dở",
    openThreadsTitle: "Chuyện còn dang dở",
    memoriesTitle: "Nolane đang nhớ gì về bạn",
    noMemories: "Chưa có ký ức cục bộ.",
    memoryKeep: "Giữ",
    memoryEdit: "Sửa",
    memorySave: "Lưu",
    memoryCancel: "Hủy",
    memoryDelete: "Quên",
    memoryDeleteConfirm: "Quên thật?",
    relationshipStageNew: "Mới quen",
    relationshipStageFamiliar: "Quen thuộc",
    relationshipStageClose: "Gần gũi",
    proactiveTitle: "Nolane có điều muốn nói",
    proactiveThreadBody: "Nolane vẫn đang nghĩ về một chuyện còn dang dở.",
    proactiveBody: "Bấm để mở. Bỏ qua thì Nolane sẽ im.",
    proactiveDismiss: "Bỏ qua",
    onboardingEyebrow: "Lần đầu gặp nhau",
    onboardingTitle: "Biến Nolane thành của riêng bạn trong ba bước nhỏ",
    onboardingBody: "Chỉ những điều cần thiết. Bạn có thể đổi lại bất cứ lúc nào.",
    onboardingNameLabel: "Nolane nên gọi bạn là gì?",
    onboardingLanguageLabel: "Ngôn ngữ giao diện",
    onboardingStyleLabel: "Nolane nên nói chuyện thế nào?",
    onboardingPrivacy: "Được lưu cục bộ cùng hồ sơ của bạn.",
    onboardingStart: "Bắt đầu trò chuyện",
  });

  const compactLocales = {
    zh: {
      off: "已关闭", on: "运行中", starting: "启动中", thinking: "思考中",
      turnOn: "启动 AI", turnOff: "关闭 AI", input: "给 Nolane 发消息…", send: "发送",
      emptyOffTitle: "Nolane 正在休息。", emptyOffBody: "想聊天时再启动 AI。",
      emptyOnTitle: "我在。", emptyOnBody: "说说你正在想的事。",
      settingsSubline: "属于你", mindOpen: "打开 Nolane 的状态",
      mindTitle: "Nolane 的内在状态", moodLabel: "情绪", activityLabel: "活动",
      activityReady: "准备倾听", activityReading: "正在接收你的消息",
      activityContext: "正在整合上下文", activityShaping: "正在形成回复",
      intentLabel: "意图", confidenceLabel: "确定度", threadsLabel: "正在关注",
      relationshipLabel: "关系", moodCalm: "平静", moodCurious: "好奇",
      moodWarm: "温暖", moodConcerned: "担心", moodPlayful: "活泼",
      moodIrritated: "烦躁", moodLow: "安静"
    },
    ja: {
      off: "オフ", on: "稼働中", starting: "起動中", thinking: "考え中",
      turnOn: "AIを起動", turnOff: "AIを停止", input: "Nolaneにメッセージ…", send: "送信",
      emptyOffTitle: "Nolaneは休んでいます。", emptyOffBody: "話したいときにAIを起動してください。",
      emptyOnTitle: "ここにいるよ。", emptyOnBody: "思っていることをそのまま話して。",
      settingsSubline: "あなたのNolane", mindOpen: "Nolaneの状態を見る",
      mindTitle: "Nolaneの内側", moodLabel: "気分", activityLabel: "活動",
      activityReady: "いつでも聞けるよ", activityReading: "メッセージを受け取っています",
      activityContext: "文脈をまとめています", activityShaping: "返答を組み立てています",
      intentLabel: "意図", confidenceLabel: "確かさ", threadsLabel: "気にしていること",
      relationshipLabel: "関係", moodCalm: "穏やか", moodCurious: "好奇心",
      moodWarm: "あたたかい", moodConcerned: "心配", moodPlayful: "遊び心",
      moodIrritated: "いら立ち", moodLow: "静か"
    },
    ko: {
      off: "꺼짐", on: "실행 중", starting: "시작 중", thinking: "생각 중",
      turnOn: "AI 켜기", turnOff: "AI 끄기", input: "Nolane에게 메시지…", send: "보내기",
      emptyOffTitle: "Nolane이 쉬고 있어요.", emptyOffBody: "대화하고 싶을 때 AI를 켜세요.",
      emptyOnTitle: "여기 있어요.", emptyOnBody: "지금 생각하는 걸 말해 주세요.",
      settingsSubline: "당신의 Nolane", mindOpen: "Nolane 상태 열기",
      mindTitle: "Nolane의 내면", moodLabel: "기분", activityLabel: "활동",
      activityReady: "들을 준비가 됐어요", activityReading: "메시지를 받아들이는 중",
      activityContext: "맥락을 모으는 중", activityShaping: "답변을 만드는 중",
      intentLabel: "의도", confidenceLabel: "확신", threadsLabel: "마음에 둔 것",
      relationshipLabel: "관계", moodCalm: "차분함", moodCurious: "호기심",
      moodWarm: "따뜻함", moodConcerned: "걱정", moodPlayful: "장난스러움",
      moodIrritated: "짜증", moodLow: "조용함"
    },
    es: {
      off: "Apagado", on: "Activo", starting: "Iniciando", thinking: "Pensando",
      turnOn: "Iniciar IA", turnOff: "Detener IA", input: "Mensaje para Nolane…", send: "Enviar",
      emptyOffTitle: "Nolane está descansando.", emptyOffBody: "Inicia la IA cuando quieras hablar.",
      emptyOnTitle: "Estoy aquí.", emptyOnBody: "Dime lo que tengas en mente.",
      settingsSubline: "tuyo", mindOpen: "Abrir estado mental de Nolane",
      mindTitle: "Dentro de Nolane", moodLabel: "Ánimo", activityLabel: "Actividad",
      activityReady: "Listo para escuchar", activityReading: "Recibiendo tu mensaje",
      activityContext: "Uniendo el contexto", activityShaping: "Formando una respuesta",
      intentLabel: "Intención", confidenceLabel: "Confianza", threadsLabel: "En mente",
      relationshipLabel: "Relación", moodCalm: "Tranquilo", moodCurious: "Curioso",
      moodWarm: "Cálido", moodConcerned: "Preocupado", moodPlayful: "Juguetón",
      moodIrritated: "Molesto", moodLow: "Sereno"
    },
    fr: {
      off: "Arrêt", on: "Actif", starting: "Démarrage", thinking: "Réflexion",
      turnOn: "Démarrer l’IA", turnOff: "Arrêter l’IA", input: "Message à Nolane…", send: "Envoyer",
      emptyOffTitle: "Nolane se repose.", emptyOffBody: "Démarrez l’IA quand vous voulez parler.",
      emptyOnTitle: "Je suis là.", emptyOnBody: "Dites ce que vous avez en tête.",
      settingsSubline: "à vous", mindOpen: "Ouvrir l’état de Nolane",
      mindTitle: "Dans Nolane", moodLabel: "Humeur", activityLabel: "Activité",
      activityReady: "Prêt à écouter", activityReading: "Lecture de votre message",
      activityContext: "Mise en contexte", activityShaping: "Construction de la réponse",
      intentLabel: "Intention", confidenceLabel: "Confiance", threadsLabel: "À l’esprit",
      relationshipLabel: "Relation", moodCalm: "Calme", moodCurious: "Curieux",
      moodWarm: "Chaleureux", moodConcerned: "Préoccupé", moodPlayful: "Joueur",
      moodIrritated: "Irrité", moodLow: "Paisible"
    },
    de: {
      off: "Aus", on: "Aktiv", starting: "Startet", thinking: "Denkt",
      turnOn: "KI starten", turnOff: "KI stoppen", input: "Nachricht an Nolane…", send: "Senden",
      emptyOffTitle: "Nolane ruht.", emptyOffBody: "Starte die KI, wenn du reden möchtest.",
      emptyOnTitle: "Ich bin da.", emptyOnBody: "Sag einfach, was dir durch den Kopf geht.",
      settingsSubline: "deins", mindOpen: "Nolanes Zustand öffnen",
      mindTitle: "In Nolane", moodLabel: "Stimmung", activityLabel: "Aktivität",
      activityReady: "Bereit zuzuhören", activityReading: "Nimmt deine Nachricht auf",
      activityContext: "Fügt Kontext zusammen", activityShaping: "Formt eine Antwort",
      intentLabel: "Absicht", confidenceLabel: "Sicherheit", threadsLabel: "Im Kopf",
      relationshipLabel: "Beziehung", moodCalm: "Ruhig", moodCurious: "Neugierig",
      moodWarm: "Warm", moodConcerned: "Besorgt", moodPlayful: "Verspielt",
      moodIrritated: "Gereizt", moodLow: "Still"
    },
    pt: {
      off: "Desligado", on: "Ativo", starting: "Iniciando", thinking: "Pensando",
      turnOn: "Iniciar IA", turnOff: "Parar IA", input: "Mensagem para Nolane…", send: "Enviar",
      emptyOffTitle: "Nolane está descansando.", emptyOffBody: "Inicie a IA quando quiser conversar.",
      emptyOnTitle: "Estou aqui.", emptyOnBody: "Diga o que estiver pensando.",
      settingsSubline: "seu", mindOpen: "Abrir estado de Nolane",
      mindTitle: "Por dentro de Nolane", moodLabel: "Humor", activityLabel: "Atividade",
      activityReady: "Pronto para ouvir", activityReading: "Recebendo sua mensagem",
      activityContext: "Reunindo contexto", activityShaping: "Formando uma resposta",
      intentLabel: "Intenção", confidenceLabel: "Confiança", threadsLabel: "Em mente",
      relationshipLabel: "Relação", moodCalm: "Calmo", moodCurious: "Curioso",
      moodWarm: "Acolhedor", moodConcerned: "Preocupado", moodPlayful: "Brincalhão",
      moodIrritated: "Irritado", moodLow: "Quieto"
    },
    it: {
      off: "Spento", on: "Attivo", starting: "Avvio", thinking: "Sta pensando",
      turnOn: "Avvia IA", turnOff: "Ferma IA", input: "Messaggio a Nolane…", send: "Invia",
      emptyOffTitle: "Nolane sta riposando.", emptyOffBody: "Avvia l’IA quando vuoi parlare.",
      emptyOnTitle: "Sono qui.", emptyOnBody: "Dimmi cosa hai in mente.",
      settingsSubline: "tuo", mindOpen: "Apri lo stato di Nolane",
      mindTitle: "Dentro Nolane", moodLabel: "Umore", activityLabel: "Attività",
      activityReady: "Pronto ad ascoltare", activityReading: "Sto ricevendo il messaggio",
      activityContext: "Sto unendo il contesto", activityShaping: "Sto formando la risposta",
      intentLabel: "Intento", confidenceLabel: "Sicurezza", threadsLabel: "In mente",
      relationshipLabel: "Relazione", moodCalm: "Calmo", moodCurious: "Curioso",
      moodWarm: "Caloroso", moodConcerned: "Preoccupato", moodPlayful: "Giocoso",
      moodIrritated: "Irritato", moodLow: "Quieto"
    },
    th: {
      off: "ปิด", on: "กำลังทำงาน", starting: "กำลังเริ่ม", thinking: "กำลังคิด",
      turnOn: "เปิด AI", turnOff: "ปิด AI", input: "ส่งข้อความถึง Nolane…", send: "ส่ง",
      emptyOffTitle: "Nolane กำลังพัก", emptyOffBody: "เปิด AI เมื่อคุณอยากคุย",
      emptyOnTitle: "ฉันอยู่นี่", emptyOnBody: "บอกสิ่งที่คุณกำลังคิดได้เลย",
      settingsSubline: "ของคุณ", mindOpen: "เปิดสถานะของ Nolane",
      mindTitle: "ภายใน Nolane", moodLabel: "อารมณ์", activityLabel: "กิจกรรม",
      activityReady: "พร้อมฟัง", activityReading: "กำลังรับข้อความของคุณ",
      activityContext: "กำลังรวมบริบท", activityShaping: "กำลังสร้างคำตอบ",
      intentLabel: "เจตนา", confidenceLabel: "ความมั่นใจ", threadsLabel: "กำลังใส่ใจ",
      relationshipLabel: "ความสัมพันธ์", moodCalm: "สงบ", moodCurious: "อยากรู้",
      moodWarm: "อบอุ่น", moodConcerned: "กังวล", moodPlayful: "ขี้เล่น",
      moodIrritated: "หงุดหงิด", moodLow: "เงียบ"
    },
    id: {
      off: "Mati", on: "Aktif", starting: "Memulai", thinking: "Berpikir",
      turnOn: "Nyalakan AI", turnOff: "Matikan AI", input: "Pesan untuk Nolane…", send: "Kirim",
      emptyOffTitle: "Nolane sedang beristirahat.", emptyOffBody: "Nyalakan AI saat ingin berbicara.",
      emptyOnTitle: "Aku di sini.", emptyOnBody: "Ceritakan apa yang ada di pikiranmu.",
      settingsSubline: "milikmu", mindOpen: "Buka keadaan Nolane",
      mindTitle: "Di dalam Nolane", moodLabel: "Suasana", activityLabel: "Aktivitas",
      activityReady: "Siap mendengarkan", activityReading: "Menerima pesanmu",
      activityContext: "Menyatukan konteks", activityShaping: "Membentuk jawaban",
      intentLabel: "Niat", confidenceLabel: "Keyakinan", threadsLabel: "Dalam pikiran",
      relationshipLabel: "Hubungan", moodCalm: "Tenang", moodCurious: "Penasaran",
      moodWarm: "Hangat", moodConcerned: "Khawatir", moodPlayful: "Ceria",
      moodIrritated: "Kesal", moodLow: "Hening"
    },
    ru: {
      off: "Выключено", on: "Работает", starting: "Запуск", thinking: "Думает",
      turnOn: "Запустить ИИ", turnOff: "Остановить ИИ", input: "Сообщение Nolane…", send: "Отправить",
      emptyOffTitle: "Nolane отдыхает.", emptyOffBody: "Запустите ИИ, когда захотите поговорить.",
      emptyOnTitle: "Я здесь.", emptyOnBody: "Расскажите, о чём думаете.",
      settingsSubline: "ваш", mindOpen: "Открыть состояние Nolane",
      mindTitle: "Внутри Nolane", moodLabel: "Настроение", activityLabel: "Активность",
      activityReady: "Готов слушать", activityReading: "Принимаю ваше сообщение",
      activityContext: "Собираю контекст", activityShaping: "Формирую ответ",
      intentLabel: "Намерение", confidenceLabel: "Уверенность", threadsLabel: "В фокусе",
      relationshipLabel: "Отношения", moodCalm: "Спокойно", moodCurious: "Любопытно",
      moodWarm: "Тепло", moodConcerned: "Обеспокоенно", moodPlayful: "Игриво",
      moodIrritated: "Раздражённо", moodLow: "Тихо"
    },
    ar: {
      off: "متوقف", on: "يعمل", starting: "يبدأ", thinking: "يفكر",
      turnOn: "تشغيل الذكاء", turnOff: "إيقاف الذكاء", input: "راسل Nolane…", send: "إرسال",
      emptyOffTitle: "Nolane يستريح.", emptyOffBody: "شغّل الذكاء عندما تريد التحدث.",
      emptyOnTitle: "أنا هنا.", emptyOnBody: "قل ما يدور في ذهنك.",
      settingsSubline: "لك", mindOpen: "فتح حالة Nolane",
      mindTitle: "داخل Nolane", moodLabel: "المزاج", activityLabel: "النشاط",
      activityReady: "جاهز للاستماع", activityReading: "يستقبل رسالتك",
      activityContext: "يجمع السياق", activityShaping: "يصوغ الرد",
      intentLabel: "النية", confidenceLabel: "الثقة", threadsLabel: "في الذهن",
      relationshipLabel: "العلاقة", moodCalm: "هادئ", moodCurious: "فضولي",
      moodWarm: "دافئ", moodConcerned: "قلق", moodPlayful: "مرح",
      moodIrritated: "منزعج", moodLow: "هادئ"
    },
    hi: {
      off: "बंद", on: "चल रहा है", starting: "शुरू हो रहा है", thinking: "सोच रहा है",
      turnOn: "AI शुरू करें", turnOff: "AI बंद करें", input: "Nolane को संदेश…", send: "भेजें",
      emptyOffTitle: "Nolane आराम कर रहा है।", emptyOffBody: "जब बात करनी हो तो AI शुरू करें।",
      emptyOnTitle: "मैं यहाँ हूँ।", emptyOnBody: "जो मन में है कहिए।",
      settingsSubline: "आपका", mindOpen: "Nolane की स्थिति खोलें",
      mindTitle: "Nolane के भीतर", moodLabel: "मूड", activityLabel: "गतिविधि",
      activityReady: "सुनने के लिए तैयार", activityReading: "आपका संदेश ले रहा है",
      activityContext: "संदर्भ जोड़ रहा है", activityShaping: "जवाब बना रहा है",
      intentLabel: "इरादा", confidenceLabel: "विश्वास", threadsLabel: "मन में",
      relationshipLabel: "रिश्ता", moodCalm: "शांत", moodCurious: "जिज्ञासु",
      moodWarm: "स्नेही", moodConcerned: "चिंतित", moodPlayful: "चंचल",
      moodIrritated: "झुंझलाया", moodLow: "शांत"
    },
    tr: {
      off: "Kapalı", on: "Çalışıyor", starting: "Başlıyor", thinking: "Düşünüyor",
      turnOn: "AI'ı başlat", turnOff: "AI'ı durdur", input: "Nolane'e mesaj…", send: "Gönder",
      emptyOffTitle: "Nolane dinleniyor.", emptyOffBody: "Konuşmak istediğinde AI'ı başlat.",
      emptyOnTitle: "Buradayım.", emptyOnBody: "Aklındakini söyle.",
      settingsSubline: "senin", mindOpen: "Nolane durumunu aç",
      mindTitle: "Nolane'in içi", moodLabel: "Ruh hali", activityLabel: "Etkinlik",
      activityReady: "Dinlemeye hazır", activityReading: "Mesajını alıyor",
      activityContext: "Bağlamı birleştiriyor", activityShaping: "Yanıtı şekillendiriyor",
      intentLabel: "Niyet", confidenceLabel: "Güven", threadsLabel: "Aklında",
      relationshipLabel: "İlişki", moodCalm: "Sakin", moodCurious: "Meraklı",
      moodWarm: "Sıcak", moodConcerned: "Endişeli", moodPlayful: "Oyuncu",
      moodIrritated: "Rahatsız", moodLow: "Sessiz"
    },
    pl: {
      off: "Wyłączone", on: "Działa", starting: "Uruchamianie", thinking: "Myśli",
      turnOn: "Uruchom AI", turnOff: "Zatrzymaj AI", input: "Wiadomość do Nolane…", send: "Wyślij",
      emptyOffTitle: "Nolane odpoczywa.", emptyOffBody: "Uruchom AI, gdy chcesz porozmawiać.",
      emptyOnTitle: "Jestem tutaj.", emptyOnBody: "Powiedz, co masz na myśli.",
      settingsSubline: "twój", mindOpen: "Otwórz stan Nolane",
      mindTitle: "Wewnątrz Nolane", moodLabel: "Nastrój", activityLabel: "Aktywność",
      activityReady: "Gotowy słuchać", activityReading: "Odbiera wiadomość",
      activityContext: "Łączy kontekst", activityShaping: "Układa odpowiedź",
      intentLabel: "Intencja", confidenceLabel: "Pewność", threadsLabel: "W pamięci",
      relationshipLabel: "Relacja", moodCalm: "Spokojny", moodCurious: "Ciekawy",
      moodWarm: "Ciepły", moodConcerned: "Zatroskany", moodPlayful: "Figlarny",
      moodIrritated: "Poirytowany", moodLow: "Cichy"
    },
    nl: {
      off: "Uit", on: "Actief", starting: "Starten", thinking: "Denkt",
      turnOn: "AI starten", turnOff: "AI stoppen", input: "Bericht aan Nolane…", send: "Verstuur",
      emptyOffTitle: "Nolane rust.", emptyOffBody: "Start de AI wanneer je wilt praten.",
      emptyOnTitle: "Ik ben er.", emptyOnBody: "Vertel wat er in je omgaat.",
      settingsSubline: "van jou", mindOpen: "Open Nolane-status",
      mindTitle: "Binnen Nolane", moodLabel: "Stemming", activityLabel: "Activiteit",
      activityReady: "Klaar om te luisteren", activityReading: "Neemt je bericht op",
      activityContext: "Brengt context samen", activityShaping: "Vormt een antwoord",
      intentLabel: "Intentie", confidenceLabel: "Zekerheid", threadsLabel: "In gedachten",
      relationshipLabel: "Relatie", moodCalm: "Kalm", moodCurious: "Nieuwsgierig",
      moodWarm: "Warm", moodConcerned: "Bezorgd", moodPlayful: "Speels",
      moodIrritated: "Geïrriteerd", moodLow: "Stil"
    }
  };
  Object.assign(strings, compactLocales);

  const v1LocaleStrings = {
    zh: {
      memoryThreadsLabel: "记忆与未完话题", openThreadsTitle: "未完话题", memoriesTitle: "Nolane 记得的事",
      noMemories: "还没有本地记忆。", memoryKeep: "保留", memoryEdit: "编辑", memorySave: "保存",
      memoryCancel: "取消", memoryDelete: "忘记", memoryDeleteConfirm: "确认忘记？",
      relationshipStageNew: "初识", relationshipStageFamiliar: "熟悉", relationshipStageClose: "亲近",
      proactiveTitle: "Nolane 有话想说", proactiveThreadBody: "Nolane 还在想着一件没聊完的事。",
      proactiveBody: "点按打开；忽略后 Nolane 会保持安静。", proactiveDismiss: "忽略",
      onboardingEyebrow: "初次见面", onboardingTitle: "用三个小步骤让 Nolane 更属于你",
      onboardingBody: "只设置必要内容，之后随时可以修改。", onboardingNameLabel: "Nolane 应该怎么称呼你？",
      onboardingLanguageLabel: "界面语言", onboardingStyleLabel: "Nolane 应该怎样和你说话？",
      onboardingPrivacy: "与个人设置一起保存在本地。", onboardingStart: "开始聊天"
    },
    ja: {
      memoryThreadsLabel: "記憶と未完の話", openThreadsTitle: "続きのある話", memoriesTitle: "Nolane が覚えていること",
      noMemories: "ローカル記憶はまだありません。", memoryKeep: "残す", memoryEdit: "編集", memorySave: "保存",
      memoryCancel: "キャンセル", memoryDelete: "忘れる", memoryDeleteConfirm: "本当に忘れる？",
      relationshipStageNew: "出会ったばかり", relationshipStageFamiliar: "慣れてきた", relationshipStageClose: "親しい",
      proactiveTitle: "Nolane が話したいことがあります", proactiveThreadBody: "まだ終わっていない話を考えています。",
      proactiveBody: "タップして開く。無視すれば Nolane は静かにしています。", proactiveDismiss: "閉じる",
      onboardingEyebrow: "はじめまして", onboardingTitle: "3つの小さな設定で Nolane をあなたらしく",
      onboardingBody: "必要なことだけ。あとでいつでも変更できます。", onboardingNameLabel: "Nolane に何と呼ばれたいですか？",
      onboardingLanguageLabel: "表示言語", onboardingStyleLabel: "Nolane の話し方は？",
      onboardingPrivacy: "プロフィールと一緒に端末内へ保存されます。", onboardingStart: "話し始める"
    },
    ko: {
      memoryThreadsLabel: "기억과 이어갈 이야기", openThreadsTitle: "끝나지 않은 이야기", memoriesTitle: "Nolane이 기억하는 것",
      noMemories: "아직 로컬 기억이 없습니다.", memoryKeep: "보관", memoryEdit: "수정", memorySave: "저장",
      memoryCancel: "취소", memoryDelete: "잊기", memoryDeleteConfirm: "정말 잊을까요?",
      relationshipStageNew: "처음", relationshipStageFamiliar: "익숙함", relationshipStageClose: "가까움",
      proactiveTitle: "Nolane이 하고 싶은 말이 있어요", proactiveThreadBody: "끝나지 않은 이야기를 계속 생각하고 있어요.",
      proactiveBody: "눌러서 열 수 있어요. 무시하면 조용히 있을게요.", proactiveDismiss: "무시",
      onboardingEyebrow: "첫 만남", onboardingTitle: "세 단계로 Nolane을 나에게 맞추기",
      onboardingBody: "꼭 필요한 것만 설정합니다. 나중에 언제든 바꿀 수 있어요.", onboardingNameLabel: "Nolane이 뭐라고 부르면 될까요?",
      onboardingLanguageLabel: "인터페이스 언어", onboardingStyleLabel: "Nolane이 어떻게 말하면 좋을까요?",
      onboardingPrivacy: "프로필과 함께 기기에 로컬 저장됩니다.", onboardingStart: "대화 시작"
    },
    es: {
      memoryThreadsLabel: "Memoria y temas pendientes", openThreadsTitle: "Temas pendientes", memoriesTitle: "Lo que Nolane recuerda",
      noMemories: "Aún no hay recuerdos locales.", memoryKeep: "Conservar", memoryEdit: "Editar", memorySave: "Guardar",
      memoryCancel: "Cancelar", memoryDelete: "Olvidar", memoryDeleteConfirm: "¿Olvidar?",
      relationshipStageNew: "Recién conocidos", relationshipStageFamiliar: "Familiar", relationshipStageClose: "Cercanos",
      proactiveTitle: "Nolane quiere decirte algo", proactiveThreadBody: "Nolane sigue pensando en algo que quedó pendiente.",
      proactiveBody: "Toca para abrir. Si lo ignoras, Nolane se queda en silencio.", proactiveDismiss: "Ignorar",
      onboardingEyebrow: "Primer encuentro", onboardingTitle: "Haz tuyo a Nolane en tres pasos",
      onboardingBody: "Solo lo esencial. Puedes cambiarlo todo después.", onboardingNameLabel: "¿Cómo quieres que te llame Nolane?",
      onboardingLanguageLabel: "Idioma de la interfaz", onboardingStyleLabel: "¿Cómo debería hablar Nolane?",
      onboardingPrivacy: "Se guarda localmente con tu perfil.", onboardingStart: "Empezar a hablar"
    },
    fr: {
      memoryThreadsLabel: "Mémoire et sujets en cours", openThreadsTitle: "Sujets en cours", memoriesTitle: "Ce que Nolane retient",
      noMemories: "Aucun souvenir local pour l’instant.", memoryKeep: "Garder", memoryEdit: "Modifier", memorySave: "Enregistrer",
      memoryCancel: "Annuler", memoryDelete: "Oublier", memoryDeleteConfirm: "Oublier ?",
      relationshipStageNew: "Nouvelle rencontre", relationshipStageFamiliar: "Familier", relationshipStageClose: "Proche",
      proactiveTitle: "Nolane a quelque chose à vous dire", proactiveThreadBody: "Nolane pense encore à un sujet resté en suspens.",
      proactiveBody: "Touchez pour ouvrir. Ignorez-le et Nolane restera discret.", proactiveDismiss: "Ignorer",
      onboardingEyebrow: "Première rencontre", onboardingTitle: "Personnalisez Nolane en trois petites étapes",
      onboardingBody: "Uniquement l’essentiel. Tout pourra être modifié ensuite.", onboardingNameLabel: "Comment Nolane doit-il vous appeler ?",
      onboardingLanguageLabel: "Langue de l’interface", onboardingStyleLabel: "Comment Nolane doit-il vous parler ?",
      onboardingPrivacy: "Enregistré localement avec votre profil.", onboardingStart: "Commencer"
    },
    de: {
      memoryThreadsLabel: "Erinnerungen & offene Themen", openThreadsTitle: "Offene Themen", memoriesTitle: "Was Nolane sich merkt",
      noMemories: "Noch keine lokalen Erinnerungen.", memoryKeep: "Behalten", memoryEdit: "Bearbeiten", memorySave: "Speichern",
      memoryCancel: "Abbrechen", memoryDelete: "Vergessen", memoryDeleteConfirm: "Wirklich vergessen?",
      relationshipStageNew: "Neu", relationshipStageFamiliar: "Vertraut", relationshipStageClose: "Nah",
      proactiveTitle: "Nolane möchte etwas sagen", proactiveThreadBody: "Nolane denkt noch über etwas Unabgeschlossenes nach.",
      proactiveBody: "Zum Öffnen tippen. Wenn du es ignorierst, bleibt Nolane still.", proactiveDismiss: "Ignorieren",
      onboardingEyebrow: "Erstes Treffen", onboardingTitle: "Mach Nolane in drei Schritten zu deinem Begleiter",
      onboardingBody: "Nur das Wesentliche. Alles lässt sich später ändern.", onboardingNameLabel: "Wie soll Nolane dich nennen?",
      onboardingLanguageLabel: "Oberflächensprache", onboardingStyleLabel: "Wie soll Nolane mit dir sprechen?",
      onboardingPrivacy: "Wird lokal mit deinem Profil gespeichert.", onboardingStart: "Gespräch starten"
    },
    pt: {
      memoryThreadsLabel: "Memória e assuntos pendentes", openThreadsTitle: "Assuntos pendentes", memoriesTitle: "O que Nolane lembra",
      noMemories: "Ainda não há memória local.", memoryKeep: "Manter", memoryEdit: "Editar", memorySave: "Salvar",
      memoryCancel: "Cancelar", memoryDelete: "Esquecer", memoryDeleteConfirm: "Esquecer mesmo?",
      relationshipStageNew: "Recém-conhecidos", relationshipStageFamiliar: "Familiar", relationshipStageClose: "Próximos",
      proactiveTitle: "Nolane quer dizer algo", proactiveThreadBody: "Nolane ainda está pensando em algo inacabado.",
      proactiveBody: "Toque para abrir. Se ignorar, Nolane fica em silêncio.", proactiveDismiss: "Ignorar",
      onboardingEyebrow: "Primeiro encontro", onboardingTitle: "Deixe Nolane com a sua cara em três passos",
      onboardingBody: "Só o essencial. Você pode mudar tudo depois.", onboardingNameLabel: "Como Nolane deve chamar você?",
      onboardingLanguageLabel: "Idioma da interface", onboardingStyleLabel: "Como Nolane deve falar?",
      onboardingPrivacy: "Salvo localmente com seu perfil.", onboardingStart: "Começar a conversar"
    },
    it: {
      memoryThreadsLabel: "Memoria e fili aperti", openThreadsTitle: "Fili aperti", memoriesTitle: "Cosa ricorda Nolane",
      noMemories: "Nessun ricordo locale per ora.", memoryKeep: "Conserva", memoryEdit: "Modifica", memorySave: "Salva",
      memoryCancel: "Annulla", memoryDelete: "Dimentica", memoryDeleteConfirm: "Dimenticare?",
      relationshipStageNew: "Appena conosciuti", relationshipStageFamiliar: "Familiari", relationshipStageClose: "Vicini",
      proactiveTitle: "Nolane vuole dirti qualcosa", proactiveThreadBody: "Nolane sta ancora pensando a qualcosa rimasto in sospeso.",
      proactiveBody: "Tocca per aprire. Se ignori, Nolane resta in silenzio.", proactiveDismiss: "Ignora",
      onboardingEyebrow: "Primo incontro", onboardingTitle: "Rendi Nolane tuo in tre piccoli passi",
      onboardingBody: "Solo l’essenziale. Potrai cambiare tutto in seguito.", onboardingNameLabel: "Come vuoi che Nolane ti chiami?",
      onboardingLanguageLabel: "Lingua dell’interfaccia", onboardingStyleLabel: "Come dovrebbe parlare Nolane?",
      onboardingPrivacy: "Salvato localmente insieme al profilo.", onboardingStart: "Inizia a parlare"
    },
    th: {
      memoryThreadsLabel: "ความทรงจำและเรื่องที่ค้างไว้", openThreadsTitle: "เรื่องที่ค้างไว้", memoriesTitle: "สิ่งที่ Nolane จำได้",
      noMemories: "ยังไม่มีความทรงจำในเครื่อง", memoryKeep: "เก็บไว้", memoryEdit: "แก้ไข", memorySave: "บันทึก",
      memoryCancel: "ยกเลิก", memoryDelete: "ลืม", memoryDeleteConfirm: "ลืมจริงไหม?",
      relationshipStageNew: "เพิ่งรู้จัก", relationshipStageFamiliar: "คุ้นเคย", relationshipStageClose: "ใกล้ชิด",
      proactiveTitle: "Nolane มีบางอย่างอยากพูด", proactiveThreadBody: "Nolane ยังคิดถึงเรื่องที่คุยกันไม่จบ",
      proactiveBody: "แตะเพื่อเปิด หากไม่สนใจ Nolane จะเงียบไว้", proactiveDismiss: "ไม่สนใจ",
      onboardingEyebrow: "พบกันครั้งแรก", onboardingTitle: "ตั้งค่า Nolane ให้เป็นของคุณในสามขั้นตอน",
      onboardingBody: "ตั้งค่าเฉพาะสิ่งจำเป็น และเปลี่ยนได้ภายหลัง", onboardingNameLabel: "อยากให้ Nolane เรียกคุณว่าอะไร?",
      onboardingLanguageLabel: "ภาษาของอินเทอร์เฟซ", onboardingStyleLabel: "อยากให้ Nolane พูดแบบไหน?",
      onboardingPrivacy: "เก็บไว้ในเครื่องพร้อมโปรไฟล์ของคุณ", onboardingStart: "เริ่มคุย"
    },
    id: {
      memoryThreadsLabel: "Memori & hal yang belum selesai", openThreadsTitle: "Hal yang belum selesai", memoriesTitle: "Yang diingat Nolane",
      noMemories: "Belum ada memori lokal.", memoryKeep: "Simpan", memoryEdit: "Edit", memorySave: "Simpan",
      memoryCancel: "Batal", memoryDelete: "Lupakan", memoryDeleteConfirm: "Yakin lupakan?",
      relationshipStageNew: "Baru kenal", relationshipStageFamiliar: "Akrab", relationshipStageClose: "Dekat",
      proactiveTitle: "Nolane ingin mengatakan sesuatu", proactiveThreadBody: "Nolane masih memikirkan hal yang belum selesai.",
      proactiveBody: "Ketuk untuk membuka. Abaikan dan Nolane akan tetap diam.", proactiveDismiss: "Abaikan",
      onboardingEyebrow: "Pertemuan pertama", onboardingTitle: "Jadikan Nolane milikmu dalam tiga langkah",
      onboardingBody: "Hanya yang penting. Semuanya bisa diubah nanti.", onboardingNameLabel: "Nolane sebaiknya memanggilmu apa?",
      onboardingLanguageLabel: "Bahasa antarmuka", onboardingStyleLabel: "Bagaimana Nolane sebaiknya berbicara?",
      onboardingPrivacy: "Disimpan lokal bersama profilmu.", onboardingStart: "Mulai mengobrol"
    },
    ru: {
      memoryThreadsLabel: "Память и незавершённые темы", openThreadsTitle: "Незавершённые темы", memoriesTitle: "Что помнит Nolane",
      noMemories: "Локальных воспоминаний пока нет.", memoryKeep: "Сохранить", memoryEdit: "Изменить", memorySave: "Сохранить",
      memoryCancel: "Отмена", memoryDelete: "Забыть", memoryDeleteConfirm: "Точно забыть?",
      relationshipStageNew: "Недавно знакомы", relationshipStageFamiliar: "Знакомы", relationshipStageClose: "Близки",
      proactiveTitle: "Nolane хочет кое-что сказать", proactiveThreadBody: "Nolane всё ещё думает о незавершённой теме.",
      proactiveBody: "Нажмите, чтобы открыть. Если проигнорировать, Nolane останется тихим.", proactiveDismiss: "Игнорировать",
      onboardingEyebrow: "Первая встреча", onboardingTitle: "Настройте Nolane под себя за три шага",
      onboardingBody: "Только самое важное. Всё можно изменить позже.", onboardingNameLabel: "Как Nolane должен вас называть?",
      onboardingLanguageLabel: "Язык интерфейса", onboardingStyleLabel: "Как Nolane должен говорить?",
      onboardingPrivacy: "Хранится локально вместе с профилем.", onboardingStart: "Начать разговор"
    },
    ar: {
      memoryThreadsLabel: "الذكريات والمواضيع المفتوحة", openThreadsTitle: "مواضيع لم تكتمل", memoriesTitle: "ما يتذكره Nolane",
      noMemories: "لا توجد ذكريات محلية بعد.", memoryKeep: "احتفاظ", memoryEdit: "تعديل", memorySave: "حفظ",
      memoryCancel: "إلغاء", memoryDelete: "نسيان", memoryDeleteConfirm: "نسيان فعلًا؟",
      relationshipStageNew: "تعارف جديد", relationshipStageFamiliar: "مألوف", relationshipStageClose: "قريب",
      proactiveTitle: "لدى Nolane شيء يريد قوله", proactiveThreadBody: "ما زال Nolane يفكر في موضوع لم يكتمل.",
      proactiveBody: "اضغط للفتح. إذا تجاهلته فسيبقى Nolane هادئًا.", proactiveDismiss: "تجاهل",
      onboardingEyebrow: "اللقاء الأول", onboardingTitle: "اجعل Nolane مناسبًا لك في ثلاث خطوات",
      onboardingBody: "الأساسيات فقط، ويمكن تغيير كل شيء لاحقًا.", onboardingNameLabel: "بماذا تريد أن يناديك Nolane؟",
      onboardingLanguageLabel: "لغة الواجهة", onboardingStyleLabel: "كيف تريد أن يتحدث Nolane؟",
      onboardingPrivacy: "يُحفظ محليًا مع ملفك الشخصي.", onboardingStart: "ابدأ الحديث"
    },
    hi: {
      memoryThreadsLabel: "यादें और अधूरी बातें", openThreadsTitle: "अधूरी बातें", memoriesTitle: "Nolane क्या याद रखता है",
      noMemories: "अभी कोई लोकल याद नहीं है।", memoryKeep: "रखें", memoryEdit: "संपादित करें", memorySave: "सहेजें",
      memoryCancel: "रद्द करें", memoryDelete: "भूलें", memoryDeleteConfirm: "सच में भूलें?",
      relationshipStageNew: "नई पहचान", relationshipStageFamiliar: "परिचित", relationshipStageClose: "करीबी",
      proactiveTitle: "Nolane कुछ कहना चाहता है", proactiveThreadBody: "Nolane अभी भी एक अधूरी बात के बारे में सोच रहा है।",
      proactiveBody: "खोलने के लिए टैप करें। अनदेखा करने पर Nolane शांत रहेगा।", proactiveDismiss: "अनदेखा",
      onboardingEyebrow: "पहली मुलाकात", onboardingTitle: "तीन छोटे चरणों में Nolane को अपना बनाएं",
      onboardingBody: "सिर्फ ज़रूरी बातें। बाद में सब बदल सकते हैं।", onboardingNameLabel: "Nolane आपको क्या कहकर बुलाए?",
      onboardingLanguageLabel: "इंटरफ़ेस भाषा", onboardingStyleLabel: "Nolane कैसे बात करे?",
      onboardingPrivacy: "आपकी प्रोफ़ाइल के साथ लोकल रूप से सहेजा जाता है।", onboardingStart: "बात शुरू करें"
    },
    tr: {
      memoryThreadsLabel: "Hafıza ve açık konular", openThreadsTitle: "Yarım kalan konular", memoriesTitle: "Nolane'ın hatırladıkları",
      noMemories: "Henüz yerel hafıza yok.", memoryKeep: "Sakla", memoryEdit: "Düzenle", memorySave: "Kaydet",
      memoryCancel: "İptal", memoryDelete: "Unut", memoryDeleteConfirm: "Gerçekten unut?",
      relationshipStageNew: "Yeni tanıştık", relationshipStageFamiliar: "Tanıdık", relationshipStageClose: "Yakın",
      proactiveTitle: "Nolane bir şey söylemek istiyor", proactiveThreadBody: "Nolane yarım kalan bir şeyi hâlâ düşünüyor.",
      proactiveBody: "Açmak için dokun. Görmezden gelirsen Nolane sessiz kalır.", proactiveDismiss: "Yoksay",
      onboardingEyebrow: "İlk buluşma", onboardingTitle: "Üç küçük adımda Nolane'ı kendine göre ayarla",
      onboardingBody: "Yalnızca gerekli olanlar. Sonra her şeyi değiştirebilirsin.", onboardingNameLabel: "Nolane sana nasıl hitap etsin?",
      onboardingLanguageLabel: "Arayüz dili", onboardingStyleLabel: "Nolane nasıl konuşsun?",
      onboardingPrivacy: "Profilinle birlikte yerel olarak saklanır.", onboardingStart: "Konuşmaya başla"
    },
    pl: {
      memoryThreadsLabel: "Pamięć i otwarte wątki", openThreadsTitle: "Niedokończone wątki", memoriesTitle: "Co pamięta Nolane",
      noMemories: "Brak lokalnych wspomnień.", memoryKeep: "Zachowaj", memoryEdit: "Edytuj", memorySave: "Zapisz",
      memoryCancel: "Anuluj", memoryDelete: "Zapomnij", memoryDeleteConfirm: "Na pewno zapomnieć?",
      relationshipStageNew: "Nowa znajomość", relationshipStageFamiliar: "Znajomi", relationshipStageClose: "Blisko",
      proactiveTitle: "Nolane chce coś powiedzieć", proactiveThreadBody: "Nolane nadal myśli o niedokończonej sprawie.",
      proactiveBody: "Dotknij, aby otworzyć. Zignoruj, a Nolane pozostanie cicho.", proactiveDismiss: "Ignoruj",
      onboardingEyebrow: "Pierwsze spotkanie", onboardingTitle: "Dopasuj Nolane do siebie w trzech krokach",
      onboardingBody: "Tylko najważniejsze rzeczy. Wszystko można później zmienić.", onboardingNameLabel: "Jak Nolane ma się do Ciebie zwracać?",
      onboardingLanguageLabel: "Język interfejsu", onboardingStyleLabel: "Jak Nolane ma mówić?",
      onboardingPrivacy: "Zapisywane lokalnie razem z profilem.", onboardingStart: "Zacznij rozmowę"
    },
    nl: {
      memoryThreadsLabel: "Geheugen en open onderwerpen", openThreadsTitle: "Open onderwerpen", memoriesTitle: "Wat Nolane onthoudt",
      noMemories: "Nog geen lokale herinneringen.", memoryKeep: "Bewaren", memoryEdit: "Bewerken", memorySave: "Opslaan",
      memoryCancel: "Annuleren", memoryDelete: "Vergeten", memoryDeleteConfirm: "Echt vergeten?",
      relationshipStageNew: "Net ontmoet", relationshipStageFamiliar: "Vertrouwd", relationshipStageClose: "Hecht",
      proactiveTitle: "Nolane wil iets zeggen", proactiveThreadBody: "Nolane denkt nog aan iets dat niet af was.",
      proactiveBody: "Tik om te openen. Negeer het en Nolane blijft stil.", proactiveDismiss: "Negeren",
      onboardingEyebrow: "Eerste ontmoeting", onboardingTitle: "Maak Nolane van jou in drie kleine stappen",
      onboardingBody: "Alleen het belangrijkste. Je kunt alles later wijzigen.", onboardingNameLabel: "Hoe moet Nolane je noemen?",
      onboardingLanguageLabel: "Interfacetaal", onboardingStyleLabel: "Hoe moet Nolane praten?",
      onboardingPrivacy: "Wordt lokaal bij je profiel opgeslagen.", onboardingStart: "Begin te praten"
    }
  };
  Object.entries(v1LocaleStrings).forEach(([code, values]) => {
    Object.assign(strings[code], values);
  });

  const UI_LOCALE_KEY = "nolane.ui.locale.v1";
  const AI_NAME_KEY = "nolane.ui.ai-name.v1";
  const AI_AVATAR_KEY = "nolane.ui.ai-avatar.v1";
  const ONBOARDING_KEY = "nolane.onboarding.v1";
  const PROACTIVE_STATE_KEY = "nolane.proactive-state.v1";
  const supportedUiLocales = new Set([
    "en", "vi", "zh", "ja", "ko", "es", "fr", "de", "pt", "it",
    "th", "id", "ru", "ar", "hi", "tr", "pl", "nl",
  ]);
  let uiLocale = (() => {
    try {
      const stored = localStorage.getItem(UI_LOCALE_KEY);
      return supportedUiLocales.has(stored) ? stored : "en";
    } catch (_error) {
      return "en";
    }
  })();

  const readStoredString = (key, fallback = "") => {
    try {
      const value = localStorage.getItem(key);
      return typeof value === "string" && value ? value : fallback;
    } catch (_error) {
      return fallback;
    }
  };

  const writeStoredString = (key, value) => {
    try {
      if (value) localStorage.setItem(key, value);
      else localStorage.removeItem(key);
    } catch (_error) {
      // Identity skin persistence is best-effort in restricted webviews.
    }
  };

  const readStoredJson = (key, fallback = {}) => {
    try {
      const value = localStorage.getItem(key);
      return value ? JSON.parse(value) : fallback;
    } catch (_error) {
      return fallback;
    }
  };

  const writeStoredJson = (key, value) => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (_error) {
      // Best-effort UI state only; runtime truth remains authoritative.
    }
  };

  let assistantName = readStoredString(AI_NAME_KEY, "Nolane")
    .replace(/[\u0000-\u001f\u007f]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 32) || "Nolane";
  let assistantAvatar = readStoredString(AI_AVATAR_KEY, "");

  let profile = {
    preferred_name: "",
    language: "auto",
    response_length: "balanced",
    conversation_style: "natural",
    initiative: "gentle",
    memory_enabled: true,
    personal_instruction: "",
  };
  let runtime = {
    phase: "off",
    powered: false,
    error: null,
  };
  let messages = [];
  let target = { mode: "browser-mock" };
  let sending = false;
  let renderedEventIds = new Set();
  let firstRender = true;
  let learningWindows = [];
  let learningAvailable = true;
  let activeLearningWindow = null;
  let learningCandidate = null;
  let thinkingStartedAt = 0;
  let thinkingTicker = null;
  let pendingProactive = null;
  let proactiveState = readStoredJson(PROACTIVE_STATE_KEY, {});

  const locale = () => uiLocale;
  const t = (key) =>
    strings[locale()]?.[key] || strings.en[key] || strings.vi[key] || key;
  const ti = (key) => String(t(key)).replaceAll("Nolane", assistantName);

  function assistantInitial() {
    return Array.from(assistantName.trim())[0]?.toUpperCase() || "N";
  }

  function applyAssistantAvatar(target) {
    if (!target) return;
    target.textContent = assistantAvatar ? "" : assistantInitial();
    target.classList.toggle("has-custom-avatar", Boolean(assistantAvatar));
    target.style.backgroundImage = assistantAvatar
      ? `url("${assistantAvatar.replaceAll('"', '%22')}")`
      : "";
  }

  function renderAssistantIdentity() {
    document.title = assistantName;
    els.assistantNameButton.textContent = assistantName;
    els.assistantNameButton.setAttribute("aria-label", t("renameAi") + ": " + assistantName);
    els.assistantNameButton.title = t("renameAi");
    els.assistantNameInput.value = assistantName;
    els.assistantNameInput.setAttribute("aria-label", t("aiName"));
    els.assistantAvatarButton.setAttribute("aria-label", t("changeAvatar"));
    els.assistantAvatarButton.title = t("changeAvatar");
    els.assistantAvatarFallback.textContent = assistantInitial();

    if (assistantAvatar) {
      els.assistantAvatarImage.src = assistantAvatar;
      els.assistantAvatarImage.hidden = false;
      els.assistantAvatarFallback.hidden = true;
    } else {
      els.assistantAvatarImage.removeAttribute("src");
      els.assistantAvatarImage.hidden = true;
      els.assistantAvatarFallback.hidden = false;
    }

    els.conversation
      .querySelectorAll(".assistant-mark")
      .forEach((node) => applyAssistantAvatar(node));
  }

  function normalizeAssistantName(value) {
    return String(value || "")
      .replace(/[\u0000-\u001f\u007f]/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, 32);
  }

  function beginAssistantNameEdit() {
    els.assistantNameButton.hidden = true;
    els.assistantNameInput.hidden = false;
    els.assistantNameInput.value = assistantName;
    els.assistantNameInput.focus();
    els.assistantNameInput.select();
  }

  function finishAssistantNameEdit({ cancel = false } = {}) {
    const next = cancel ? assistantName : normalizeAssistantName(els.assistantNameInput.value);
    if (!cancel && next) {
      assistantName = next;
      writeStoredString(AI_NAME_KEY, assistantName === "Nolane" ? "" : assistantName);
    }
    els.assistantNameInput.hidden = true;
    els.assistantNameButton.hidden = false;
    renderAssistantIdentity();
    applyLocale();
  }

  function readImageAsDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(reader.error || new Error("Could not read image"));
      reader.onload = () => resolve(String(reader.result || ""));
      reader.readAsDataURL(file);
    });
  }

  async function prepareAvatar(file) {
    if (!file || !String(file.type || "").startsWith("image/")) {
      throw new Error("Avatar must be an image");
    }
    if (file.size > 12 * 1024 * 1024) {
      throw new Error("Avatar image is too large");
    }
    const source = await readImageAsDataUrl(file);
    const image = new Image();
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error("Could not decode avatar"));
      image.src = source;
    });
    const size = 256;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Canvas is unavailable");
    const side = Math.min(image.naturalWidth, image.naturalHeight);
    const sx = Math.max(0, (image.naturalWidth - side) / 2);
    const sy = Math.max(0, (image.naturalHeight - side) / 2);
    context.drawImage(image, sx, sy, side, side, 0, 0, size, size);
    return canvas.toDataURL("image/webp", 0.86);
  }

  const hasTauri = () =>
    Boolean(window.__TAURI__ && window.__TAURI__.core?.invoke);

  const invoke = async (command, args = {}) => {
    if (!hasTauri()) {
      throw new Error("Tauri bridge unavailable");
    }
    return window.__TAURI__.core.invoke(command, args);
  };

  const mock = {
    status: {
      schema: "NOLANE-PRODUCT-RUNTIME-STATUS-V1",
      phase: "off",
      powered: false,
      error: null,
      identity_id: "browser-preview",
      state_version: 0,
      interactions: 0,
      open_threads: 0,
      memory_enabled: true,
      initiative: "gentle",
      model_checkpoint_sha256: "preview",
      device: "preview",
      mind: {
        schema: "NOLANE-OBSERVABLE-MIND-V1",
        raw_reasoning_exposed: false,
        affect: {
          valence: 0.12,
          energy: 0.68,
          playfulness: 0.48,
          irritation: 0.02,
          concern: 0.08,
          social_drive: 0.24,
        },
        working: {
          curiosity: 0.58,
          uncertainty: 0.18,
          active_intent: "conversation",
          recent_topics: [],
        },
        relationship: {
          closeness: 0.16,
          trust: 0.18,
          familiarity: 0.10,
        },
        open_threads: [],
        memories: [
          {
            id: "preview-memory-1",
            text: "Bạn thích câu trả lời ngắn và tự nhiên.",
            kind: "preference",
            confidence: 0.88,
            salience: 0.82,
            kept: false,
          },
        ],
      },
    },
    profile: { ...profile, digest: "preview" },
    messages: [],
    learningWindows: [],
    learningCandidates: {},
  };

  async function mockApi(method, path, body) {
    if (method === "GET" && path === "/v1/status") {
      return { ...mock.status };
    }
    if (method === "GET" && path.startsWith("/v1/history")) {
      return { messages: [...mock.messages] };
    }
    if (method === "GET" && path === "/v1/profile") {
      return { ...mock.profile };
    }
    if (method === "PUT" && path === "/v1/profile") {
      mock.profile = { ...mock.profile, ...body };
      mock.status.memory_enabled = Boolean(mock.profile.memory_enabled);
      mock.status.initiative = mock.profile.initiative;
      return { ...mock.profile };
    }
    if (method === "POST" && path === "/v1/power") {
      mock.status.phase = body.enabled ? "on" : "off";
      mock.status.powered = Boolean(body.enabled);
      return { ...mock.status };
    }
    if (method === "POST" && path === "/v1/memory") {
      const rows = mock.status.mind.memories || [];
      const index = rows.findIndex((row) => row.id === body.memory_id);
      if (index < 0) throw new Error("unknown memory");
      if (body.action === "keep") {
        rows[index].kept = true;
        const [memory] = rows.splice(index, 1);
        rows.unshift(memory);
      } else if (body.action === "edit") {
        const text = String(body.text || "").trim();
        if (!text) throw new Error("memory text is empty");
        rows[index].text = text.slice(0, 4000);
      } else if (body.action === "delete") {
        rows.splice(index, 1);
      } else {
        throw new Error("unsupported memory action");
      }
      return { mind: { ...mock.status.mind, memories: rows } };
    }
    if (method === "POST" && path === "/v1/chat") {
      if (!mock.status.powered) {
        throw new Error("Nolane AI is not running");
      }
      const now = new Date().toISOString();
      mock.messages.push({
        event_id: "preview-u-" + mock.messages.length,
        at: now,
        role: "user",
        text: body.text,
      });
      mock.messages.push({
        event_id: "preview-a-" + mock.messages.length,
        at: now,
        role: "assistant",
        text:
          locale() === "vi"
            ? "Mình đang ở đây. Bản preview này chỉ kiểm tra giao diện; khi chạy trong app, câu trả lời sẽ đến từ runtime Nolane thật."
            : "I'm here. This browser preview only verifies the interface; in the app, replies come from the real Nolane runtime.",
      });
      return { reply: mock.messages.at(-1).text };
    }
    if (method === "GET" && path === "/v1/learning/windows") {
      return { windows: mock.learningWindows.map((row) => ({ ...row })) };
    }
    if (method === "POST" && path === "/v1/learning/windows") {
      const pending = mock.learningWindows.find((row) => !row.intake_ready);
      if (pending) {
        throw new Error("A review window is already pending");
      }
      const index = mock.learningWindows.length + 1;
      const windowId = "window-" + String(index).padStart(4, "0");
      const candidates = [
        {
          candidate_id: windowId + "-c1",
          prompt: "Mình thích câu trả lời ngắn và thẳng hơn.",
          target: "Được, mình sẽ ưu tiên trả lời gọn và đi thẳng vào ý chính.",
          language: "vi",
        },
        {
          candidate_id: windowId + "-c2",
          prompt: "Khi mình hỏi code, hãy đưa ví dụ cụ thể.",
          target: "Ừ, mình sẽ kèm ví dụ chạy được khi nó giúp câu trả lời rõ hơn.",
          language: "vi",
        },
        {
          candidate_id: windowId + "-c3",
          prompt: "Remind me to avoid generic filler.",
          target: "I will keep answers concrete and avoid generic filler.",
          language: "en",
        },
      ];
      mock.learningCandidates[windowId] = candidates;
      const row = {
        window_id: windowId,
        phase: "QUEUE_READY",
        review_progress: {
          total: candidates.length,
          decided: 0,
          approved_non_sensitive: 0,
          rejected: 0,
          sensitive: 0,
          remaining: candidates.length,
        },
        intake_ready: false,
        approved_manifest_sha256: null,
        quality_status: null,
        quality_court_sha256: null,
        through_rowid_inclusive: index * 100,
        authority: "BROWSER_PREVIEW_ONLY",
      };
      mock.learningWindows.push(row);
      return { ...row };
    }
    if (
      method === "GET" &&
      path.startsWith("/v1/learning/pending?window_id=")
    ) {
      const parsed = new URL(path, "http://mock.local");
      const windowId = parsed.searchParams.get("window_id");
      const row = mock.learningWindows.find((item) => item.window_id === windowId);
      if (!row) throw new Error("unknown product evidence window");
      const candidates = mock.learningCandidates[windowId] || [];
      const candidate = candidates[row.review_progress.decided] || null;
      return {
        candidate: candidate
          ? {
              window_id: windowId,
              ...candidate,
              progress: { ...row.review_progress },
              privacy: {
                raw_text_returned_to_local_authenticated_ui: true,
                network_model_call: false,
              },
            }
          : null,
      };
    }
    if (method === "POST" && path === "/v1/learning/decision") {
      const row = mock.learningWindows.find(
        (item) => item.window_id === body.window_id,
      );
      if (!row) throw new Error("unknown product evidence window");
      const candidates = mock.learningCandidates[row.window_id] || [];
      const expected = candidates[row.review_progress.decided];
      if (!expected || expected.candidate_id !== body.candidate_id) {
        throw new Error("candidate already decided or out of order");
      }
      row.review_progress.decided += 1;
      row.review_progress.remaining -= 1;
      if (body.decision === "approve") {
        row.review_progress.approved_non_sensitive += 1;
      } else {
        row.review_progress.rejected += 1;
        if (body.decision === "sensitive") {
          row.review_progress.sensitive += 1;
        }
      }
      row.phase =
        row.review_progress.remaining === 0
          ? "REVIEW_COMPLETE"
          : "REVIEW_IN_PROGRESS";
      const next = candidates[row.review_progress.decided] || null;
      return {
        window: { ...row, review_progress: { ...row.review_progress } },
        next_candidate: next
          ? {
              window_id: row.window_id,
              ...next,
              progress: { ...row.review_progress },
            }
          : null,
      };
    }
    if (method === "POST" && path === "/v1/learning/finalize") {
      const row = mock.learningWindows.find(
        (item) => item.window_id === body.window_id,
      );
      if (!row) throw new Error("unknown product evidence window");
      if (row.review_progress.remaining !== 0) {
        throw new Error("review candidates remain undecided");
      }
      row.phase = "INTAKE_READY";
      row.intake_ready = true;
      row.approved_manifest_sha256 = "a".repeat(64);
      row.quality_status = "PASS";
      row.quality_court_sha256 = "b".repeat(64);
      return { ...row, review_progress: { ...row.review_progress } };
    }
    throw new Error("Unsupported mock route " + method + " " + path);
  }

  async function api(method, path, body = null) {
    if (!hasTauri()) {
      return mockApi(method, path, body);
    }
    const result = await invoke("product_api", {
      method,
      path,
      body,
    });
    if (typeof result === "string") {
      return JSON.parse(result);
    }
    return result;
  }

  async function loadTarget() {
    if (!hasTauri()) {
      target = { mode: "browser-mock", endpoint: "mock://local" };
      els.connectionDetails.hidden = true;
      return;
    }
    target = await invoke("runtime_target");
    els.connectionDetails.hidden = ["local", "local-mobile"].includes(target.mode);
    els.remoteEndpoint.value = target.endpoint || "";
  }

  function setText(id, key) {
    const node = $(id);
    if (node) node.textContent = t(key);
  }

  function applyLocale() {
    document.documentElement.lang = locale();
    document.documentElement.dir = locale() === "ar" ? "rtl" : "ltr";
    if (els.uiLanguage) els.uiLanguage.value = locale();

    els.profileButton.setAttribute("aria-label", t("profileAria"));
    els.conversation.setAttribute("aria-label", ti("conversationAria"));
    els.powerButtonLabel.textContent =
      runtime.phase === "on" || runtime.phase === "thinking"
        ? t("turnOff")
        : t("turnOn");
    els.input.placeholder = ti("input");
    els.send.setAttribute("aria-label", t("send"));
    els.retry.textContent = t("retry");
    els.identitySubline.textContent = t("settingsSubline");
    if (els.mindStateButton) {
      els.mindStateButton.setAttribute("aria-label", ti("mindOpen"));
    }

    setText("settingsEyebrow", "settingsEyebrow");
    $("settingsTitle").textContent = ti("settingsTitle");
    setText("preferredNameLabel", "preferredNameLabel");
    setText("languageLabel", "responseLanguageLabel");
    setText("lengthLabel", "lengthLabel");
    setText("styleLabel", "styleLabel");
    setText("initiativeLabel", "initiativeLabel");
    setText("memoryTitle", "memoryTitle");
    setText("memoryBody", "memoryBody");
    setText("instructionLabel", "instructionLabel");
    setText("saveProfileButton", "saveProfile");
    if (els.preferredName) els.preferredName.placeholder = t("preferredNamePlaceholder");

    setText("mindEyebrow", "mindEyebrow");
    $("mindTitle").textContent = ti("mindTitle");
    setText("mindDisclaimer", "mindDisclaimer");
    setText("mindMoodLabel", "moodLabel");
    setText("mindActivityLabel", "activityLabel");
    setText("mindActivityHint", "activityHint");
    setText("mindIntentLabel", "intentLabel");
    setText("mindConfidenceLabel", "confidenceLabel");
    setText("mindConfidenceHint", "confidenceHint");
    setText("mindThreadsLabel", "memoryThreadsLabel");
    setText("openThreadsTitle", "openThreadsTitle");
    setText("memoriesTitle", "memoriesTitle");
    setText("mindRelationshipLabel", "relationshipLabel");
    setText("mindClosenessLabel", "closeness");
    setText("mindTrustLabel", "trust");
    setText("mindFamiliarityLabel", "familiarity");
    setText("onboardingEyebrow", "onboardingEyebrow");
    $("onboardingTitle").textContent = ti("onboardingTitle");
    setText("onboardingBody", "onboardingBody");
    $("onboardingNameLabel").textContent = ti("onboardingNameLabel");
    setText("onboardingLanguageLabel", "onboardingLanguageLabel");
    $("onboardingStyleLabel").textContent = ti("onboardingStyleLabel");
    setText("onboardingPrivacy", "onboardingPrivacy");
    setText("onboardingStart", "onboardingStart");
    if (els.onboardingLocale) els.onboardingLocale.value = locale();
    if (els.proactiveDismiss) {
      els.proactiveDismiss.setAttribute("aria-label", t("proactiveDismiss"));
      els.proactiveDismiss.title = t("proactiveDismiss");
    }

    renderAssistantIdentity();
    renderRuntime();
    renderMind();
    renderThinking();
    renderProactive();
  }

  function phaseForUi() {
    if (sending && runtime.phase === "on") return "thinking";
    return runtime.phase || "off";
  }

  function renderRuntime() {
    const phase = phaseForUi();
    els.powerButton.dataset.phase = phase;
    els.powerButton.setAttribute("aria-checked", String(runtime.phase === "on"));
    els.powerButton.setAttribute(
      "aria-label",
      runtime.phase === "on" ? t("turnOff") : t("turnOn"),
    );
    els.powerButtonLabel.textContent =
      runtime.phase === "on" || phase === "thinking" ? t("turnOff") : t("turnOn");

    const statusText = {
      off: t("off"),
      starting: t("starting"),
      on: t("on"),
      thinking: t("thinking"),
      error: t("error"),
    }[phase] || t("off");
    els.powerStatusText.textContent = statusText;

    const enabled = runtime.phase === "on" && !sending;
    els.input.disabled = !enabled;
    els.send.disabled = !enabled || !els.input.value.trim();
    els.powerButton.disabled = runtime.phase === "starting" || sending;

    if (runtime.error) {
      showBanner(humanizeRuntimeError(runtime.error), true);
    } else if (target.mode === "unconfigured") {
      showBanner(ti("remoteMissing"), false);
    } else {
      hideBanner();
    }

    if (messages.length === 0) {
      els.emptyState.hidden = false;
      if (runtime.phase === "on") {
        els.emptyTitle.textContent = ti("emptyOnTitle");
        els.emptyBody.textContent = ti("emptyOnBody");
      } else {
        els.emptyTitle.textContent = ti("emptyOffTitle");
        els.emptyBody.textContent = ti("emptyOffBody");
      }
    } else {
      els.emptyState.hidden = true;
    }
    renderMind();
  }

  function humanizeRuntimeError(error) {
    const value = String(error || "");
    if (
      value.includes("checkpoint") ||
      value.includes("tokenizer") ||
      value.includes("release model")
    ) {
      return t("modelMissing");
    }
    return value || t("unavailable");
  }

  function showBanner(message, retry = true) {
    els.banner.hidden = false;
    els.bannerText.textContent = message;
    els.retry.hidden = !retry;
  }

  function hideBanner() {
    els.banner.hidden = true;
    els.retry.hidden = true;
  }

  function nearBottom() {
    const el = els.conversation;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 130;
  }

  function scrollBottom(force = false) {
    if (force || nearBottom()) {
      requestAnimationFrame(() => {
        els.conversation.scrollTop = els.conversation.scrollHeight;
      });
    }
  }

  function renderMessage(message) {
    const node = document.createElement("article");
    node.className =
      "message " + (message.role === "user" ? "user" : "assistant");
    node.dataset.eventId = message.event_id || "";
    if (message.role !== "user") {
      const mark = document.createElement("div");
      mark.className = "assistant-mark";
      mark.setAttribute("aria-hidden", "true");
      applyAssistantAvatar(mark);
      node.appendChild(mark);
    }
    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = message.text;
    node.appendChild(content);
    return node;
  }

  function renderMessages(nextMessages) {
    const shouldFollow = firstRender || nearBottom();
    const nextIds = new Set(
      nextMessages.map((m, index) =>
        m.event_id || [m.role, index, m.text].join("-"),
      ),
    );

    if (
      firstRender ||
      nextMessages.length < messages.length ||
      [...renderedEventIds].some((id) => !nextIds.has(id))
    ) {
      els.conversation
        .querySelectorAll(".message, .thinking-row")
        .forEach((node) => node.remove());
      renderedEventIds = new Set();
    }

    nextMessages.forEach((message, index) => {
      const id =
        message.event_id || [message.role, index, message.text].join("-");
      if (renderedEventIds.has(id)) return;
      els.conversation.appendChild(renderMessage(message));
      renderedEventIds.add(id);
    });

    messages = nextMessages;
    firstRender = false;
    renderThinking();
    renderRuntime();
    if (shouldFollow) scrollBottom(true);
  }

  const clamp01 = (value, fallback = 0) => {
    const number = Number(value);
    return Number.isFinite(number) ? Math.max(0, Math.min(1, number)) : fallback;
  };

  function mindSnapshot() {
    const mind = runtime.mind || {};
    const affect = mind.affect || {};
    const working = mind.working || {};
    const relationship = mind.relationship || {};
    return {
      valence: Math.max(-1, Math.min(1, Number(affect.valence) || 0)),
      energy: clamp01(affect.energy, 0.62),
      playfulness: clamp01(affect.playfulness, 0.45),
      irritation: clamp01(affect.irritation, 0),
      concern: clamp01(affect.concern, 0),
      curiosity: clamp01(working.curiosity ?? runtime.lifecycle?.curiosity, 0.35),
      uncertainty: working.uncertainty == null
        ? null
        : clamp01(working.uncertainty, 0),
      intent: working.active_intent || null,
      threads: Array.isArray(mind.open_threads)
        ? mind.open_threads.filter(Boolean).slice(0, 3)
        : [],
      memories: Array.isArray(mind.memories)
        ? mind.memories.filter((row) => row && row.text).slice(0, 5)
        : [],
      recentTopics: Array.isArray(working.recent_topics)
        ? working.recent_topics.filter(Boolean).slice(-3)
        : [],
      closeness: clamp01(relationship.closeness, 0.05),
      trust: clamp01(relationship.trust, 0.05),
      familiarity: clamp01(relationship.familiarity, 0),
    };
  }

  function moodFromMind(snapshot) {
    if (snapshot.irritation >= 0.48) return "Irritated";
    if (snapshot.concern >= 0.5) return "Concerned";
    if (snapshot.valence <= -0.32) return "Low";
    if (snapshot.playfulness >= 0.66 && snapshot.valence > 0.05) return "Playful";
    if (snapshot.valence >= 0.28 && snapshot.concern < 0.35) return "Warm";
    if (snapshot.curiosity >= 0.6) return "Curious";
    return "Calm";
  }

  function moodText(mood) {
    return t({
      Irritated: "moodIrritated",
      Concerned: "moodConcerned",
      Low: "moodLow",
      Playful: "moodPlayful",
      Warm: "moodWarm",
      Curious: "moodCurious",
      Calm: "moodCalm",
    }[mood] || "moodCalm");
  }

  function visibleActivity() {
    if (!sending) return { key: "activityReady", progress: 12 };
    const elapsed = Math.max(0, Date.now() - thinkingStartedAt);
    if (elapsed < 1400) return { key: "activityReading", progress: 30 };
    if (elapsed < 3600) return { key: "activityContext", progress: 58 };
    return { key: "activityShaping", progress: 82 };
  }

  function relationshipStage(snapshot) {
    const score =
      0.45 * snapshot.familiarity +
      0.35 * snapshot.closeness +
      0.20 * snapshot.trust;
    if (score >= 0.55) return t("relationshipStageClose");
    if (score >= 0.22) return t("relationshipStageFamiliar");
    return t("relationshipStageNew");
  }

  async function controlMemory(action, memoryId, text = null) {
    try {
      const result = await api("POST", "/v1/memory", {
        action,
        memory_id: memoryId,
        text,
      });
      if (result?.mind) {
        runtime = { ...runtime, mind: result.mind };
      } else {
        await refreshStatus();
      }
      renderMind();
    } catch (error) {
      showBanner(String(error?.message || error), false);
    }
  }

  function memoryActionButton(label, action, memory) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "memory-action";
    button.textContent = label;
    button.addEventListener("click", () => {
      if (action === "keep") void controlMemory("keep", memory.id);
    });
    return button;
  }

  function renderMemory(memory) {
    const item = document.createElement("article");
    item.className = "mind-memory";
    if (memory.kept) item.dataset.kept = "true";

    const text = document.createElement("p");
    text.className = "mind-memory-text";
    text.textContent = memory.text;

    const actions = document.createElement("div");
    actions.className = "memory-actions";

    const keep = memoryActionButton(
      (memory.kept ? "✓ " : "") + t("memoryKeep"),
      "keep",
      memory,
    );
    keep.disabled = Boolean(memory.kept);

    const edit = document.createElement("button");
    edit.type = "button";
    edit.className = "memory-action";
    edit.textContent = t("memoryEdit");

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "memory-action memory-danger";
    remove.textContent = t("memoryDelete");
    let deleteArmed = false;
    let deleteTimer = null;
    remove.addEventListener("click", () => {
      if (!deleteArmed) {
        deleteArmed = true;
        remove.textContent = t("memoryDeleteConfirm");
        deleteTimer = setTimeout(() => {
          deleteArmed = false;
          remove.textContent = t("memoryDelete");
        }, 2600);
        return;
      }
      if (deleteTimer) clearTimeout(deleteTimer);
      void controlMemory("delete", memory.id);
    });

    actions.append(keep, edit, remove);
    item.append(text, actions);

    edit.addEventListener("click", () => {
      if (item.querySelector(".memory-editor")) return;
      const editor = document.createElement("div");
      editor.className = "memory-editor";
      const input = document.createElement("textarea");
      input.rows = 2;
      input.maxLength = 4000;
      input.value = memory.text;
      const editorActions = document.createElement("div");
      editorActions.className = "memory-editor-actions";
      const cancel = document.createElement("button");
      cancel.type = "button";
      cancel.className = "memory-action";
      cancel.textContent = t("memoryCancel");
      const save = document.createElement("button");
      save.type = "button";
      save.className = "memory-action memory-primary";
      save.textContent = t("memorySave");
      cancel.addEventListener("click", () => editor.remove());
      save.addEventListener("click", () => {
        const next = input.value.trim();
        if (!next) return;
        void controlMemory("edit", memory.id, next);
      });
      editorActions.append(cancel, save);
      editor.append(input, editorActions);
      item.append(editor);
      input.focus();
      input.select();
    });

    return item;
  }

  function renderMind() {
    if (!els.mindStateButton) return;
    const powered = runtime.phase === "on" || sending;
    els.mindStateButton.hidden = !powered;
    if (!powered) return;

    const snapshot = mindSnapshot();
    const mood = moodFromMind(snapshot);
    const moodLabel = moodText(mood);
    els.mindStateLabel.textContent = moodLabel;
    els.mindStateButton.dataset.mood = mood.toLowerCase();
    els.mindOrb.dataset.mood = mood.toLowerCase();
    els.mindHeroOrb.dataset.mood = mood.toLowerCase();
    els.mindMood.textContent = moodLabel;
    if (els.appShell) {
      els.appShell.dataset.mood = mood.toLowerCase();
      els.appShell.dataset.presence =
        sending ? "thinking" : runtime.phase === "on" ? "present" : "resting";
    }

    const activity = visibleActivity();
    els.mindActivity.textContent = t(activity.key);
    els.mindProgressBar.style.width = activity.progress + "%";
    els.mindIntent.textContent =
      snapshot.intent && snapshot.intent !== "conversation"
        ? snapshot.intent.replaceAll("_", " ")
        : t(sending ? "intentReply" : "intentConversation");

    els.mindConfidence.textContent =
      snapshot.uncertainty == null
        ? t("confidenceSteady")
        : snapshot.uncertainty <= 0.32
          ? t("confidenceClear")
          : snapshot.uncertainty <= 0.62
            ? t("confidenceSteady")
            : t("confidenceUncertain");

    const chips = [
      ["emotionCuriosity", snapshot.curiosity],
      ["emotionConcern", snapshot.concern],
      ["emotionPlayful", snapshot.playfulness],
      ["emotionEnergy", snapshot.energy],
      ["emotionIrritation", snapshot.irritation],
    ]
      .filter(([, value]) => value >= 0.34)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);
    els.emotionChips.replaceChildren();
    (chips.length ? chips : [["emotionEnergy", snapshot.energy]]).forEach(
      ([key, value]) => {
        const chip = document.createElement("span");
        chip.className = "emotion-chip";
        chip.textContent = t(key) + " · " + Math.round(value * 100) + "%";
        els.emotionChips.appendChild(chip);
      },
    );

    const threads = snapshot.threads.length
      ? snapshot.threads
      : snapshot.recentTopics;
    els.mindThreads.replaceChildren();
    if (!threads.length) {
      const empty = document.createElement("span");
      empty.className = "mind-empty";
      empty.textContent = t("noThreads");
      els.mindThreads.appendChild(empty);
    } else {
      threads.forEach((topic) => {
        const item = document.createElement("span");
        item.className = "mind-thread";
        item.textContent = topic;
        els.mindThreads.appendChild(item);
      });
    }

    els.mindMemories.replaceChildren();
    if (!snapshot.memories.length) {
      const empty = document.createElement("span");
      empty.className = "mind-empty";
      empty.textContent = t("noMemories");
      els.mindMemories.appendChild(empty);
    } else {
      snapshot.memories.forEach((memory) => {
        els.mindMemories.appendChild(renderMemory(memory));
      });
    }

    els.mindRelationshipStage.textContent = relationshipStage(snapshot);
    els.closenessBar.style.width = Math.round(snapshot.closeness * 100) + "%";
    els.trustBar.style.width = Math.round(snapshot.trust * 100) + "%";
    els.familiarityBar.style.width = Math.round(snapshot.familiarity * 100) + "%";
  }

  function ensureThinkingTicker() {
    if (!sending) {
      if (thinkingTicker) clearInterval(thinkingTicker);
      thinkingTicker = null;
      return;
    }
    if (thinkingTicker) return;
    thinkingTicker = setInterval(() => {
      renderThinking();
      renderMind();
    }, 700);
  }

  function renderThinking() {
    els.conversation.querySelectorAll(".thinking-row").forEach((n) => n.remove());
    ensureThinkingTicker();
    renderMind();
    if (!sending) return;

    const activity = visibleActivity();
    const snapshot = mindSnapshot();
    const mood = moodFromMind(snapshot);
    const row = document.createElement("div");
    row.className = "thinking-row";

    const button = document.createElement("button");
    button.type = "button";
    button.className = "thinking-capsule";
    button.dataset.mood = mood.toLowerCase();
    button.setAttribute("aria-label", t("mindOpen"));

    const orb = document.createElement("span");
    orb.className = "thinking-orb";
    orb.setAttribute("aria-hidden", "true");

    const copy = document.createElement("span");
    copy.className = "thinking-copy";
    const title = document.createElement("strong");
    title.textContent = t("thinking");
    const detail = document.createElement("small");
    detail.textContent = t(activity.key);
    copy.append(title, detail);

    const dots = document.createElement("span");
    dots.className = "thinking-dots";
    dots.setAttribute("aria-hidden", "true");
    dots.innerHTML = "<i></i><i></i><i></i>";

    button.append(orb, copy, dots);
    button.addEventListener("click", () => {
      renderMind();
      if (!els.mindDialog.open) els.mindDialog.showModal();
    });
    row.append(button);
    els.conversation.appendChild(row);
    scrollBottom(true);
  }

  async function refreshStatus() {
    try {
      runtime = await api("GET", "/v1/status");
      renderRuntime();
      renderMind();
    } catch (error) {
      runtime = {
        ...runtime,
        phase: "error",
        powered: false,
        error: String(error?.message || error),
      };
      renderRuntime();
    }
  }

  function proactiveId(message) {
    return String(
      message?.event_id ||
      [message?.at || "", message?.intent || "", message?.text || ""].join(":"),
    );
  }

  function isProactiveMessage(message) {
    if (!message || message.role === "user") return false;
    const eventId = String(message.event_id || "");
    const intent = String(message.intent || "");
    return (
      eventId.startsWith("local-mobile-i-") ||
      Boolean(intent && intent !== "respond_to_user" && intent !== "conversation")
    );
  }

  function rememberProactive(id, action) {
    proactiveState[id] = action;
    const entries = Object.entries(proactiveState);
    if (entries.length > 80) {
      proactiveState = Object.fromEntries(entries.slice(-60));
    }
    writeStoredJson(PROACTIVE_STATE_KEY, proactiveState);
  }

  function renderProactive() {
    if (!els.proactiveWrap) return;
    els.proactiveWrap.hidden = !pendingProactive;
    if (!pendingProactive) return;
    els.proactiveTitle.textContent = ti("proactiveTitle");
    const intent = String(pendingProactive.intent || "");
    els.proactiveBody.textContent = intent.startsWith("follow_up:")
      ? ti("proactiveThreadBody")
      : t("proactiveBody");
    const orb = els.proactiveWrap.querySelector(".proactive-orb");
    if (orb) orb.dataset.mood = moodFromMind(mindSnapshot()).toLowerCase();
  }

  function filterProactiveMessages(nextMessages) {
    const rows = Array.isArray(nextMessages) ? nextMessages : [];
    pendingProactive = null;
    const visible = [];
    for (const message of rows) {
      if (!isProactiveMessage(message)) {
        visible.push(message);
        continue;
      }
      const id = proactiveId(message);
      const state = proactiveState[id];
      if (state === "open") {
        visible.push(message);
      } else if (state !== "dismiss") {
        pendingProactive = message;
      }
    }
    renderProactive();
    return visible;
  }

  async function refreshHistory() {
    try {
      const result = await api("GET", "/v1/history");
      renderMessages(filterProactiveMessages(result.messages));
    } catch (error) {
      showBanner(String(error?.message || error), true);
    }
  }

  async function refreshProfile() {
    try {
      profile = await api("GET", "/v1/profile");
      fillProfile();
      applyLocale();
    } catch (error) {
      showBanner(String(error?.message || error), true);
    }
  }

  function pendingLearningWindow() {
    return [...learningWindows]
      .reverse()
      .find((row) => !row.intake_ready) || null;
  }

  function renderLearningSummary() {
    if (!learningAvailable) {
      els.learningSummary.textContent = t("learningUnavailable");
      els.prepareLearningWindow.disabled = true;
      els.openLearningReview.disabled = true;
      return;
    }

    const pending = pendingLearningWindow();
    const ready = learningWindows.filter((row) => row.intake_ready).length;
    if (pending) {
      const progress = pending.review_progress || {};
      els.learningSummary.textContent =
        String(progress.decided || 0) +
        "/" +
        String(progress.total || 0) +
        " " +
        t("learningReviewed");
      els.prepareLearningWindow.disabled = true;
      els.openLearningReview.disabled = false;
      return;
    }

    els.prepareLearningWindow.disabled = false;
    els.openLearningReview.disabled = true;
    els.learningSummary.textContent =
      ready > 0
        ? String(ready) + " " + t("learningReady")
        : t("learningNone");
  }

  async function refreshLearning() {
    if (target.mode === "unconfigured") {
      learningAvailable = false;
      learningWindows = [];
      renderLearningSummary();
      return;
    }
    try {
      const result = await api("GET", "/v1/learning/windows");
      learningWindows = Array.isArray(result.windows) ? result.windows : [];
      learningAvailable = true;
    } catch (_error) {
      learningAvailable = false;
      learningWindows = [];
    }
    renderLearningSummary();
  }

  function updateLearningApprovalState() {
    els.approveLearning.disabled =
      !learningCandidate || !els.reviewTarget.value.trim();
  }

  function renderLearningCandidate(candidate) {
    learningCandidate = candidate || null;
    if (els.learningBody) {
      els.learningBody.scrollTop = 0;
    }
    const window = activeLearningWindow;
    const progress = candidate?.progress || window?.review_progress || {};
    els.learningProgress.textContent =
      String(progress.decided || 0) +
      "/" +
      String(progress.total || 0) +
      " " +
      t("learningReviewed");

    if (!candidate) {
      els.reviewCandidate.hidden = true;
      els.reviewComplete.hidden = false;
      els.reviewActions.hidden = true;
      els.finalizeLearning.hidden = false;
      updateLearningApprovalState();
      return;
    }

    els.reviewCandidate.hidden = false;
    els.reviewComplete.hidden = true;
    els.reviewActions.hidden = false;
    els.finalizeLearning.hidden = true;
    els.reviewPrompt.textContent = candidate.prompt || "";
    els.reviewTarget.value = candidate.target || "";
    updateLearningApprovalState();
    const language =
      candidate.language === "vi" || candidate.language === "en"
        ? candidate.language
        : profile.language === "vi" || profile.language === "en"
          ? profile.language
          : locale();
    els.reviewLanguage.value = language === "en" ? "en" : "vi";
  }

  async function fetchLearningCandidate(windowId) {
    const result = await api(
      "GET",
      "/v1/learning/pending?window_id=" + encodeURIComponent(windowId),
    );
    renderLearningCandidate(result.candidate || null);
  }

  async function openLearningDialog() {
    const pending = pendingLearningWindow();
    if (!pending) return;
    activeLearningWindow = pending;
    els.dialog.close();
    els.learningDialog.showModal();
    try {
      await fetchLearningCandidate(pending.window_id);
    } catch (error) {
      els.learningDialog.close();
      showBanner(String(error?.message || error), false);
    }
  }

  async function prepareLearningReview() {
    if (!learningAvailable || pendingLearningWindow()) return;
    els.learningSummary.textContent = t("learningPreparing");
    els.prepareLearningWindow.disabled = true;
    try {
      const window = await api("POST", "/v1/learning/windows", {});
      await refreshLearning();
      activeLearningWindow =
        learningWindows.find((row) => row.window_id === window.window_id) ||
        pendingLearningWindow();
      if (activeLearningWindow) {
        els.dialog.close();
        els.learningDialog.showModal();
        await fetchLearningCandidate(activeLearningWindow.window_id);
      }
    } catch (error) {
      await refreshLearning();
      const message = String(error?.message || error);
      showBanner(
        message.includes("3 leakage-safe")
          ? t("learningNeedMore")
          : message,
        false,
      );
    }
  }

  async function submitLearningDecision(decision) {
    if (!activeLearningWindow || !learningCandidate) return;
    for (const button of [
      els.approveLearning,
      els.rejectLearning,
      els.markSensitive,
    ]) {
      button.disabled = true;
    }
    try {
      const result = await api("POST", "/v1/learning/decision", {
        window_id: activeLearningWindow.window_id,
        candidate_id: learningCandidate.candidate_id,
        decision,
        language: els.reviewLanguage.value,
        weight: 1.0,
        corrected_target:
          decision === "approve" &&
          els.reviewTarget.value.trim() !==
            String(learningCandidate.target || "").trim()
            ? els.reviewTarget.value.trim()
            : null,
      });
      activeLearningWindow = result.window;
      learningWindows = learningWindows.map((row) =>
        row.window_id === result.window.window_id ? result.window : row,
      );
      renderLearningCandidate(result.next_candidate || null);
      renderLearningSummary();
    } catch (error) {
      showBanner(String(error?.message || error), false);
    } finally {
      els.rejectLearning.disabled = false;
      els.markSensitive.disabled = false;
      updateLearningApprovalState();
    }
  }

  async function finalizeLearningWindow() {
    if (!activeLearningWindow) return;
    els.finalizeLearning.disabled = true;
    try {
      await api("POST", "/v1/learning/finalize", {
        window_id: activeLearningWindow.window_id,
      });
      await refreshLearning();
      activeLearningWindow = null;
      learningCandidate = null;
      els.learningDialog.close();
      els.dialog.showModal();
    } catch (error) {
      showBanner(String(error?.message || error), false);
    } finally {
      els.finalizeLearning.disabled = false;
    }
  }

  async function togglePower() {
    const next = runtime.phase !== "on";
    runtime = { ...runtime, phase: next ? "starting" : "off", error: null };
    renderRuntime();
    try {
      runtime = await api("POST", "/v1/power", { enabled: next });
      renderRuntime();
      if (runtime.phase === "on") {
        els.input.focus();
      }
    } catch (error) {
      runtime = {
        ...runtime,
        phase: "error",
        powered: false,
        error: String(error?.message || error),
      };
      renderRuntime();
    }
  }

  function autoResize() {
    els.input.style.height = "auto";
    els.input.style.height = Math.min(els.input.scrollHeight, 148) + "px";
    els.send.disabled =
      runtime.phase !== "on" || sending || !els.input.value.trim();
  }

  async function sendMessage() {
    const text = els.input.value.trim();
    if (!text || sending || runtime.phase !== "on") return;
    sending = true;
    thinkingStartedAt = Date.now();
    const pending = {
      event_id: "pending-" + Date.now(),
      at: new Date().toISOString(),
      role: "user",
      text,
    };
    renderMessages([...messages, pending]);
    els.input.value = "";
    autoResize();
    renderRuntime();

    try {
      await api("POST", "/v1/chat", { text });
      await refreshHistory();
    } catch (error) {
      showBanner(String(error?.message || error), true);
      await refreshHistory();
    } finally {
      sending = false;
      thinkingStartedAt = 0;
      renderThinking();
      await refreshStatus();
      renderRuntime();
      els.input.focus();
    }
  }

  function fillProfile() {
    els.preferredName.value = profile.preferred_name || "";
    els.language.value = profile.language || "auto";
    els.responseLength.value = profile.response_length || "balanced";
    els.conversationStyle.value = profile.conversation_style || "natural";
    els.initiative.value = profile.initiative || "gentle";
    els.memoryEnabled.checked = profile.memory_enabled !== false;
    els.personalInstruction.value = profile.personal_instruction || "";
  }

  function profilePatch() {
    return {
      preferred_name: els.preferredName.value,
      language: els.language.value,
      response_length: els.responseLength.value,
      conversation_style: els.conversationStyle.value,
      initiative: els.initiative.value,
      memory_enabled: els.memoryEnabled.checked,
      personal_instruction: els.personalInstruction.value,
    };
  }

  async function saveProfile() {
    els.saveState.textContent = "";
    try {
      profile = await api("PUT", "/v1/profile", profilePatch());
      fillProfile();
      applyLocale();
      els.saveState.textContent = t("saved");
      setTimeout(() => {
        if (els.saveState.textContent === t("saved")) {
          els.saveState.textContent = "";
        }
      }, 1400);
    } catch (error) {
      els.saveState.textContent = t("saveError");
      showBanner(String(error?.message || error), true);
    }
  }

  async function saveConnection() {
    if (!hasTauri()) return;
    const endpoint = els.remoteEndpoint.value.trim();
    const token = els.remoteToken.value;
    try {
      target = await invoke("configure_remote", { endpoint, token });
      els.remoteToken.value = "";
      els.saveState.textContent = t("connectionSaved");
      await refreshAll();
    } catch (error) {
      els.saveState.textContent = t("saveError");
      showBanner(String(error?.message || error), false);
    }
  }

  async function clearConnection() {
    if (!hasTauri()) return;
    try {
      target = await invoke("clear_remote");
      els.remoteEndpoint.value = "";
      els.remoteToken.value = "";
      els.saveState.textContent = t("connectionCleared");
      await refreshAll();
    } catch (error) {
      showBanner(String(error?.message || error), false);
    }
  }

  function onboardingComplete() {
    return readStoredString(ONBOARDING_KEY, "") === "complete";
  }

  function maybeShowOnboarding() {
    if (!els.onboardingDialog || onboardingComplete()) return;
    els.onboardingName.value = profile.preferred_name || "";
    els.onboardingLocale.value = locale();
    els.onboardingStyle.value = profile.conversation_style || "natural";
    if (!els.onboardingDialog.open) {
      els.onboardingDialog.showModal();
      requestAnimationFrame(() => els.onboardingName.focus());
    }
  }

  async function finishOnboarding(event) {
    event.preventDefault();
    const nextLocale = els.onboardingLocale.value;
    if (supportedUiLocales.has(nextLocale)) {
      uiLocale = nextLocale;
      writeStoredString(UI_LOCALE_KEY, uiLocale);
    }
    try {
      profile = await api("PUT", "/v1/profile", {
        ...profile,
        preferred_name: els.onboardingName.value.trim(),
        conversation_style: els.onboardingStyle.value,
      });
      writeStoredString(ONBOARDING_KEY, "complete");
      fillProfile();
      applyLocale();
      els.onboardingDialog.close();
      if (runtime.phase === "on") els.input.focus();
    } catch (error) {
      showBanner(String(error?.message || error), false);
    }
  }

  async function refreshAll() {
    await loadTarget();
    if (target.mode === "unconfigured") {
      runtime = {
        phase: "error",
        powered: false,
        error: null,
      };
      renderRuntime();
      return;
    }
    await Promise.all([
      refreshStatus(),
      refreshHistory(),
      refreshProfile(),
      refreshLearning(),
    ]);
    maybeShowOnboarding();
  }

  els.proactiveCapsule.addEventListener("click", () => {
    if (!pendingProactive) return;
    const message = pendingProactive;
    rememberProactive(proactiveId(message), "open");
    pendingProactive = null;
    renderProactive();
    renderMessages([...messages, message]);
    scrollBottom(true);
  });

  els.proactiveDismiss.addEventListener("click", () => {
    if (!pendingProactive) return;
    rememberProactive(proactiveId(pendingProactive), "dismiss");
    pendingProactive = null;
    renderProactive();
  });

  els.onboardingForm.addEventListener("submit", (event) => {
    void finishOnboarding(event);
  });

  els.onboardingLocale.addEventListener("change", () => {
    const next = els.onboardingLocale.value;
    if (!supportedUiLocales.has(next)) return;
    uiLocale = next;
    applyLocale();
  });

  els.assistantAvatarButton.addEventListener("click", () => {
    els.assistantAvatarInput.click();
  });

  els.assistantAvatarInput.addEventListener("change", async () => {
    const [file] = els.assistantAvatarInput.files || [];
    els.assistantAvatarInput.value = "";
    if (!file) return;
    try {
      assistantAvatar = await prepareAvatar(file);
      writeStoredString(AI_AVATAR_KEY, assistantAvatar);
      renderAssistantIdentity();
    } catch (error) {
      showBanner(String(error?.message || error), false);
    }
  });

  els.assistantNameButton.addEventListener("click", beginAssistantNameEdit);
  els.assistantNameInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      finishAssistantNameEdit();
    } else if (event.key === "Escape") {
      event.preventDefault();
      finishAssistantNameEdit({ cancel: true });
    }
  });
  els.assistantNameInput.addEventListener("blur", () => {
    if (!els.assistantNameInput.hidden) finishAssistantNameEdit();
  });

  els.uiLanguage.addEventListener("change", () => {
    const next = els.uiLanguage.value;
    uiLocale = supportedUiLocales.has(next) ? next : "en";
    try {
      localStorage.setItem(UI_LOCALE_KEY, uiLocale);
    } catch (_error) {
      // Local persistence is best-effort in restricted browser contexts.
    }
    applyLocale();
  });

  els.mindStateButton.addEventListener("click", () => {
    renderMind();
    if (!els.mindDialog.open) els.mindDialog.showModal();
  });
  els.closeMind.addEventListener("click", () => els.mindDialog.close());
  els.mindDialog.addEventListener("click", (event) => {
    const rect = els.mindDialog.getBoundingClientRect();
    const outside =
      event.clientX < rect.left ||
      event.clientX > rect.right ||
      event.clientY < rect.top ||
      event.clientY > rect.bottom;
    if (outside) els.mindDialog.close();
  });

  els.powerButton.addEventListener("click", togglePower);
  els.retry.addEventListener("click", refreshAll);

  els.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    void sendMessage();
  });

  els.input.addEventListener("input", autoResize);
  els.input.addEventListener("keydown", (event) => {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.isComposing &&
      window.innerWidth > 640
    ) {
      event.preventDefault();
      void sendMessage();
    }
  });

  els.profileButton.addEventListener("click", () => {
    fillProfile();
    els.saveState.textContent = "";
    els.dialog.showModal();
    requestAnimationFrame(() => els.preferredName.focus());
  });

  els.closeProfile.addEventListener("click", () => els.dialog.close());

  els.profileForm.addEventListener("submit", (event) => {
    event.preventDefault();
    void saveProfile().then(() => els.dialog.close());
  });

  els.saveConnection.addEventListener("click", () => void saveConnection());
  els.clearConnection.addEventListener("click", () => void clearConnection());

  els.prepareLearningWindow.addEventListener(
    "click",
    () => void prepareLearningReview(),
  );
  els.openLearningReview.addEventListener(
    "click",
    () => void openLearningDialog(),
  );
  els.closeLearning.addEventListener(
    "click",
    () => els.learningDialog.close(),
  );
  els.approveLearning.addEventListener(
    "click",
    () => void submitLearningDecision("approve"),
  );
  els.rejectLearning.addEventListener(
    "click",
    () => void submitLearningDecision("reject"),
  );
  els.markSensitive.addEventListener(
    "click",
    () => void submitLearningDecision("sensitive"),
  );
  els.finalizeLearning.addEventListener(
    "click",
    () => void finalizeLearningWindow(),
  );
  els.reviewTarget.addEventListener(
    "input",
    updateLearningApprovalState,
  );

  els.dialog.addEventListener("click", (event) => {
    const rect = els.dialog.getBoundingClientRect();
    const outside =
      event.clientX < rect.left ||
      event.clientX > rect.right ||
      event.clientY < rect.top ||
      event.clientY > rect.bottom;
    if (outside) els.dialog.close();
  });


  window.addEventListener("focus", () => void refreshAll());

  autoResize();
  els.uiLanguage.value = uiLocale;
  renderAssistantIdentity();
  applyLocale();
  void refreshAll();

  setInterval(() => {
    if (!document.hidden && !sending && target.mode !== "unconfigured") {
      void refreshStatus();
      void refreshHistory();
    }
  }, 3000);
})();
