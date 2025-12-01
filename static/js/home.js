const steps = [
  {
    id: "setup",
    title: "1. Preparação",
    badge: "Conectar AEDT",
    commands: [
      {
        id: "list-projects",
        title: "Listar projetos .aedt",
        desc: "Consulta o diretório do servidor e retorna projetos disponíveis.",
        actions: [
          "GET /api/projects",
          "Validar caminho base e permissões",
          "Exibir lista na UI para seleção",
        ],
      },
      {
        id: "open-project",
        title: "Abrir projeto",
        desc: "Carrega o projeto no PyAEDT e mantém sessão ativa.",
        actions: [
          "POST /api/projects/open { path }",
          "Inicializar aplicação AEDT/HFSS headless quando possível",
          "Retornar designs e setups do projeto selecionado",
        ],
      },
      {
        id: "session-status",
        title: "Status da sessão",
        desc: "Confere se a instância PyAEDT está saudável.",
        actions: [
          "GET /api/session/health",
          "Retornar versão AEDT, host e PID",
          "Notificar falhas ou necessidade de restart",
        ],
      },
    ],
  },
  {
    id: "design",
    title: "2. Design e setup",
    badge: "Configurar simulação",
    commands: [
      {
        id: "list-designs",
        title: "Listar designs",
        desc: "Recupera designs HFSS, Circuit, etc. disponíveis no projeto.",
        actions: [
          "GET /api/projects/{project}/designs",
          "Selecionar design ativo",
          "Carregar setups e sweeps associados",
        ],
      },
      {
        id: "list-setups",
        title: "Setups e sweeps",
        desc: "Exibe setups, sweeps e parâmetros configuráveis.",
        actions: [
          "GET /api/designs/{design}/setups",
          "GET /api/designs/{design}/sweeps",
          "Permitir seleção de sweep e porta",
        ],
      },
      {
        id: "run-analysis",
        title: "Executar análise",
        desc: "Dispara simulação ou reutiliza resultados cacheados.",
        actions: [
          "POST /api/designs/{design}/analyze { setup, sweep }",
          "Monitorar progresso de job pesado",
          "Retornar caminho de resultados e relatórios criados",
        ],
      },
    ],
  },
  {
    id: "reports",
    title: "3. Relatórios e dados",
    badge: "Resultados",
    commands: [
      {
        id: "sparams",
        title: "Extrair S-parameters",
        desc: "Gera S11/Snm e exporta CSV/Touchstone.",
        actions: [
          "POST /api/designs/{design}/reports/sparams { sweep, ports }",
          "Exportar Touchstone/CSV e mini estatísticas",
          "Responder metadados para Plotly no front-end",
        ],
      },
      {
        id: "radiation",
        title: "Diagramas de radiação",
        desc: "Extrai HRP/VRP e vistas 3D renderizadas.",
        actions: [
          "POST /api/designs/{design}/reports/radiation { cut, freq }",
          "Gerar PNG/CSV resumidos",
          "Retornar métricas (HPBW, SLL, F/B)",
        ],
      },
      {
        id: "export-datasheet",
        title: "Exportar datasheet",
        desc: "Gera PDF consolidado com métricas e gráficos.",
        actions: [
          "POST /api/exports/datasheet { project, design, theme }",
          "Persistir artefatos em /exports",
          "Devolver URL de download e checksum",
        ],
      },
    ],
  },
  {
    id: "ai",
    title: "4. IA (Gemini)",
    badge: "Análises",
    commands: [
      {
        id: "ai-analyze",
        title: "Analisar resultado atual",
        desc: "Envio resumido (sem arrays enormes) para o Gemini.",
        actions: [
          "POST /api/ai/analyze { context, resumo }",
          "Exibir resposta em #ai-output",
          "Registrar tokens usados para auditoria",
        ],
      },
      {
        id: "ai-suggest",
        title: "Sugerir parâmetros",
        desc: "Sugere ajustes de sweep/porta para otimização.",
        actions: [
          "POST /api/ai/suggest-params { parameter_name, values, target }",
          "Exibir sugestões aplicáveis na UI",
          "Guardar histórico para futura comparação",
        ],
      },
    ],
  },
];

const state = {
  selectedStep: null,
  selectedCommand: null,
  stepStatus: {},
};

function el(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text) element.textContent = text;
  return element;
}

function renderSteps() {
  const container = document.getElementById("steps-container");
  container.innerHTML = "";
  steps.forEach((step) => {
    const card = el("div", "step-card");
    const header = el("div", "step-header");
    const title = el("div", "step-title");
    title.textContent = step.title;
    const badge = el("div", "step-badge", step.badge);
    header.append(title, badge);
    card.append(header);

    const list = el("div", "command-list");
    step.commands.forEach((command) => {
      const cmd = el("div", "command");
      cmd.dataset.stepId = step.id;
      cmd.dataset.commandId = command.id;
      const cmdTitle = el("div", "command-title", command.title);
      const cmdDesc = el("div", "command-desc", command.desc);
      cmd.append(cmdTitle, cmdDesc);
      cmd.addEventListener("click", () => selectCommand(step, command));
      list.append(cmd);
    });

    card.append(list);
    container.append(card);
  });
}

function selectCommand(step, command) {
  state.selectedStep = step.id;
  state.selectedCommand = command.id;

  const subtitle = document.getElementById("command-subtitle");
  subtitle.textContent = `${step.title} • ${command.title}`;

  const body = document.getElementById("command-body");
  body.innerHTML = "";

  const desc = el("p");
  desc.textContent = command.desc;
  body.append(desc);

  const actionsTitle = el("div", "panel-subtitle", "Sequência sugerida:");
  body.append(actionsTitle);

  const list = document.createElement("ol");
  command.actions.forEach((action) => {
    const li = document.createElement("li");
    li.textContent = action;
    list.append(li);
  });
  body.append(list);

  pushLog(`Comando selecionado: ${step.title} › ${command.title}`);
  updateStatus(`Pronto para executar "${command.title}"`);
}

function pushLog(message) {
  const logBody = document.getElementById("log-body");
  const entry = el("div", "log-entry");
  const now = new Date().toLocaleTimeString();
  entry.textContent = `[${now}] ${message}`;
  logBody.prepend(entry);
}

function updateStatus(text) {
  const pill = document.getElementById("status-pill");
  pill.textContent = text;
}

function handleMarkDone() {
  if (!state.selectedStep) {
    pushLog("Nenhuma etapa selecionada.");
    return;
  }
  state.stepStatus[state.selectedStep] = "done";
  pushLog(`Etapa "${state.selectedStep}" marcada como concluída.`);
  updateStatus("Etapa concluída");
}

function bindActions() {
  document.getElementById("btn-refresh")?.addEventListener("click", () => {
    renderSteps();
    pushLog("Layout recarregado.");
    updateStatus("Layout recarregado");
  });

  document.getElementById("btn-mark-done")?.addEventListener("click", handleMarkDone);

  document.getElementById("btn-clear-log")?.addEventListener("click", () => {
    document.getElementById("log-body").innerHTML = "";
    pushLog("Log limpo.");
  });
}

document.addEventListener("DOMContentLoaded", () => {
  renderSteps();
  bindActions();
  pushLog("SPA carregada.");
});
