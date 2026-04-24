#!/usr/bin/env node
/**
 * ApexVision-Core — Dev orchestrator
 * Compatible con Windows (.venv) y Linux/Mac
 */

const { spawn } = require("child_process");
const path      = require("path");
const fs        = require("fs");

const isWin   = process.platform === "win32";
const root    = path.resolve(__dirname, "..");
const venvBin = path.join(root, ".venv", isWin ? "Scripts" : "bin");

// Verificar venv
if (!fs.existsSync(venvBin)) {
  console.error(`\n❌ .venv no encontrado en: ${venvBin}`);
  console.error("   Crea el entorno: python -m venv .venv");
  console.error("   Instala deps:    pip install -r python/requirements.txt\n");
  process.exit(1);
}

// En Windows usamos shell:true y quoted paths para evitar EINVAL
const python = isWin
  ? `"${path.join(venvBin, "python.exe")}"`
  : path.join(venvBin, "python");

const celery = isWin
  ? `"${path.join(venvBin, "celery.exe")}"`
  : path.join(venvBin, "celery");

const npx = isWin ? "npx.cmd" : "npx";

const ENV = {
  ...process.env,
  PYTHONPATH: root,
  PYTHONUTF8: "1",
};

const services = [
  {
    name:  "API",
    color: "\x1b[36m",
    cmd:   `${python} -m python.main`,
  },
  {
    name:  "WORKER",
    color: "\x1b[33m",
    cmd:   `${celery} -A python.celery_app worker -l info -Q vision,batch,default --pool solo`,
  },
  {
    name:  "BEAT",
    color: "\x1b[32m",
    cmd:   `${celery} -A python.celery_app beat -l info`,
  },
  {
    name:  "TS",
    color: "\x1b[35m",
    cmd:   isWin ? `npx.cmd tsx watch src/index.ts` : `npx tsx watch src/index.ts`,
  },
];

// Filtros por flag
const argv = process.argv.slice(2);
const active = argv.includes("--api-only")
  ? services.filter(s => ["API", "WORKER"].includes(s.name))
  : argv.includes("--no-ts")
  ? services.filter(s => s.name !== "TS")
  : services;

const reset = "\x1b[0m\x1b[22m";
console.log(`\n\x1b[1m🚀 ApexVision-Core — Dev Server${reset}`);
console.log("─".repeat(50));
active.forEach(s => console.log(`  ${s.color}[${s.name}]${reset} ${s.cmd}`));
console.log("─".repeat(50) + "\n");

const procs = active.map(svc => {
  const proc = spawn(svc.cmd, [], {
    cwd:   root,
    shell: true,          // shell:true en todos — evita EINVAL en Windows
    stdio: "pipe",
    env:   ENV,
  });

  const prefix = `${svc.color}[${svc.name}]${reset} `;

  proc.stdout.on("data", d =>
    d.toString().split(/\r?\n/).filter(Boolean)
      .forEach(l => process.stdout.write(`${prefix}${l}\n`))
  );
  proc.stderr.on("data", d =>
    d.toString().split(/\r?\n/).filter(Boolean)
      .forEach(l => process.stderr.write(`${prefix}${l}\n`))
  );
  proc.on("exit", (code, sig) => {
    if (sig === "SIGINT" || sig === "SIGTERM" || code === null) return;
    console.error(`\n${prefix}salió con código ${code}`);
    if (["API", "WORKER"].includes(svc.name) && code !== 0) {
      console.error(`${prefix}Servicio crítico caído — cerrando todo...`);
      cleanup(); process.exit(code ?? 1);
    }
  });
  proc.on("error", err =>
    console.error(`${prefix}Error: ${err.message}`)
  );
  return proc;
});

function cleanup() {
  procs.forEach(p => { try { p.kill(); } catch {} });
}

process.on("SIGINT",  () => { console.log("\n⛔ Cerrando..."); cleanup(); process.exit(0); });
process.on("SIGTERM", () => { cleanup(); process.exit(0); });