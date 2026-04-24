import { describe, it, expect } from "vitest";
import { ApexVisionClient, ApexVisionError } from "../../src/sdk/apexvision";

describe("ApexVisionClient", () => {
  it("should instantiate correctly", () => {
    const client = new ApexVisionClient({ baseUrl: "http://localhost:8000", apiKey: "test" });
    expect(client).toBeDefined();
  });
  it("ApexVisionError should carry statusCode", () => {
    const err = new ApexVisionError("not found", 404, { detail: "x" });
    expect(err.statusCode).toBe(404);
    expect(err.name).toBe("ApexVisionError");
  });
});
