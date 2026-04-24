/** ApexVision-Core — TS Configuration */
export const config = {
  apiUrl:   process.env.APEX_API_URL  ?? "http://localhost:8000",
  apiKey:   process.env.APEX_API_KEY  ?? process.env.MASTER_API_KEY ?? "",
  redisUrl: process.env.REDIS_URL     ?? "redis://localhost:6379",
  port:     Number(process.env.TS_PORT ?? 3000),
} as const;
