/** ApexVision-Core — BullMQ Vision Job Worker */
import { Worker, Job } from "bullmq";
import { config } from "../config";
import { log } from "../utils/logger";

export const visionWorker = new Worker(
  "vision-jobs",
  async (job: Job) => {
    log.info(`Processing job ${job.id}: ${job.name}`);
    // Forward payload to Python FastAPI
    const res = await fetch(`${config.apiUrl}/api/v1/vision/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-ApexVision-Key": config.apiKey,
      },
      body: JSON.stringify(job.data),
    });
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
  },
  { connection: { url: config.redisUrl } }
);

visionWorker.on("completed", (job) => log.info(`Job ${job.id} completed`));
visionWorker.on("failed", (job, err) => log.error(`Job ${job?.id} failed:`, err));
