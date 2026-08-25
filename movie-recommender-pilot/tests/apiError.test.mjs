import assert from "node:assert/strict";
import test from "node:test";

import { apiErrorMessage } from "../src/utils/apiError.mjs";

test("API string details are displayed directly", () => {
  const error = { response: { data: { detail: "Model is unavailable" } } };
  assert.equal(apiErrorMessage(error, "Fallback"), "Model is unavailable");
});

test("FastAPI validation arrays become safe text", () => {
  const error = {
    response: {
      data: {
        detail: [
          {
            loc: ["body", "min_release_year"],
            msg: "Extra inputs are not permitted",
            type: "extra_forbidden",
          },
        ],
      },
    },
  };

  assert.equal(
    apiErrorMessage(error, "Fallback"),
    "min_release_year: Extra inputs are not permitted"
  );
});

test("unknown error payloads use the fallback", () => {
  assert.equal(apiErrorMessage(new Error("network"), "Try again"), "Try again");
});
