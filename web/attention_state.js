(function (global) {
  "use strict";

  const STORAGE_KEY = "busy38-management-seen-events";
  const MAX_EVENT_IDS = 250;

  function normalizeSeenState(value) {
    const payload = value && typeof value === "object" ? value : {};
    const ids = Array.isArray(payload.event_ids)
      ? payload.event_ids.map((item) => `${item || ""}`.trim()).filter(Boolean)
      : [];
    const cardReviews = payload.card_reviews && typeof payload.card_reviews === "object"
      ? Object.fromEntries(
        Object.entries(payload.card_reviews)
          .map(([cardId, reviewedAt]) => [`${cardId || ""}`.trim(), `${reviewedAt || ""}`.trim()])
          .filter(([cardId, reviewedAt]) => cardId && reviewedAt)
      )
      : {};
    return {
      event_ids: Array.from(new Set(ids)).slice(-MAX_EVENT_IDS),
      card_reviews: cardReviews,
      updated_at: `${payload.updated_at || ""}`.trim(),
    };
  }

  function readSeenState(storage) {
    if (!storage || typeof storage.getItem !== "function") {
      return normalizeSeenState(null);
    }
    try {
      const raw = storage.getItem(STORAGE_KEY);
      if (!raw) {
        return normalizeSeenState(null);
      }
      return normalizeSeenState(JSON.parse(raw));
    } catch (_) {
      return normalizeSeenState(null);
    }
  }

  function writeSeenState(storage, state) {
    const normalized = normalizeSeenState(state);
    if (!storage || typeof storage.setItem !== "function") {
      return normalized;
    }
    try {
      storage.setItem(STORAGE_KEY, JSON.stringify(normalized));
    } catch (_) {
      return normalized;
    }
    return normalized;
  }

  function markEventsSeen(storage, state, events) {
    const normalized = normalizeSeenState(state);
    const eventIds = Array.isArray(events)
      ? events.map((event) => `${event?.id || ""}`.trim()).filter(Boolean)
      : [];
    const mergedIds = Array.from(new Set([...normalized.event_ids, ...eventIds])).slice(-MAX_EVENT_IDS);
    return writeSeenState(storage, {
      ...normalized,
      event_ids: mergedIds,
      updated_at: new Date().toISOString(),
    });
  }

  function isEventSeen(state, eventId) {
    const normalized = normalizeSeenState(state);
    const target = `${eventId || ""}`.trim();
    if (!target) {
      return false;
    }
    return normalized.event_ids.includes(target);
  }

  function markCardReviewed(storage, state, cardId, reviewedAt) {
    const normalized = normalizeSeenState(state);
    const target = `${cardId || ""}`.trim();
    if (!target) {
      return normalized;
    }
    return writeSeenState(storage, {
      ...normalized,
      card_reviews: {
        ...(normalized.card_reviews || {}),
        [target]: `${reviewedAt || new Date().toISOString()}`.trim() || new Date().toISOString(),
      },
      updated_at: new Date().toISOString(),
    });
  }

  function getCardReviewedAt(state, cardId) {
    const normalized = normalizeSeenState(state);
    const target = `${cardId || ""}`.trim();
    if (!target) {
      return "";
    }
    return `${normalized.card_reviews?.[target] || ""}`.trim();
  }

  global.Busy38AttentionState = {
    STORAGE_KEY,
    normalizeSeenState,
    readSeenState,
    writeSeenState,
    markEventsSeen,
    isEventSeen,
    markCardReviewed,
    getCardReviewedAt,
  };
})(window);
