import { useMemo } from "react";
import axios from "axios";

import { PILOT_API_URL } from "../utils/apiConfig";
import {
  PROTOCOL_ID,
  PROTOCOL_VERSION,
  inputSignature,
  newStudyId,
} from "./studyProtocol.mjs";

export function useStudySession(system) {
  const participantId = useMemo(
    () => durableId("localStorage", "tears-gers-participant", "participant"),
    []
  );
  const sessionId = useMemo(
    () => durableId("sessionStorage", "tears-gers-session", "session"),
    []
  );

  const metadata = async ({
    taskId,
    input,
    representationRevision,
    requestId = newStudyId("request"),
    trialId = null,
    attempt = null,
    targetMovieId = null,
  }) => {
    return {
      protocol_id: PROTOCOL_ID,
      protocol_version: PROTOCOL_VERSION,
      participant_id: participantId,
      session_id: sessionId,
      system,
      task_id: taskId,
      trial_id: trialId,
      attempt,
      representation_revision: representationRevision,
      request_id: requestId,
      input_signature: await inputSignature(input),
      target_movie_id: targetMovieId,
    };
  };

  const persistRender = async (study, recommendations) => {
    await axios.post(`${PILOT_API_URL}/study/render`, {
      recommendations: recommendations.map((movie) => ({
        movie_id: Number(movie.movie_id),
        rank: Number(movie.rank),
        score: Number(movie.score),
      })),
      study,
    });
  };

  return {
    participantId,
    sessionId,
    metadata,
    persistRender,
  };
}

function durableId(storageName, key, prefix) {
  try {
    const storage = window[storageName];
    const existing = storage?.getItem(key);
    if (existing) return existing;
    const created = newStudyId(prefix);
    storage?.setItem(key, created);
    return created;
  } catch {
    return newStudyId(prefix);
  }
}
