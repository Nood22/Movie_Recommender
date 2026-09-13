import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import axios from "axios";
import TearsApp from "./TearsApp";
import reportedResponse from "../../../artifacts/recommendation_render_fix_20260909/reported_response.json";

// Render the full 100-title onboarding UI, not a reduced fixture catalog.
jest.setTimeout(30000);

jest.mock("axios", () => ({ post: jest.fn() }));
jest.mock("framer-motion", () => {
  const React = require("react");
  const element = (tag) => ({ whileHover, children, ...props }) =>
    React.createElement(tag, props, children);
  return { motion: { div: element("div"), button: element("button"), img: element("img") } };
});
const mockPersistRender = jest.fn().mockResolvedValue(undefined);
jest.mock("../study/useStudySession", () => ({
  useStudySession: () => ({
    metadata: async ({ taskId }) => ({ request_id: taskId, input_signature: taskId }),
    persistRender: mockPersistRender,
  }),
}));
jest.mock("../utils/tmdbMetadata.mjs", () => ({
  ...jest.requireActual("../utils/tmdbMetadata.mjs"),
  // Missing poster metadata must never remove a recommendation card.
  resolveVerifiedTMDBMetadata: jest.fn().mockResolvedValue(null),
}));

test("renders all returned movies, including unselected onboarding sequels, in the persisted order", async () => {
  mockPersistRender.mockClear();
  axios.post.mockImplementation(async (url, payload) => ({ data: url.endsWith("/summarize")
    ? { summary: "Summary: The viewer may enjoy epic fantasy adventures.",
        summary_source_request_id: "source", study: payload.study }
    : { items: reportedResponse.items.slice(0, payload.top_k), study: payload.study }
  }));
  render(<TearsApp goBack={() => {}} />);
  const title = "Lord of the Rings: The Fellowship of the Ring, The (2001)";
  fireEvent.click((await screen.findAllByText(title))[0].closest("button"));
  fireEvent.change(screen.getByLabelText(`Rating for ${title}`),
    { target: { value: "5" } });
  await waitFor(() => expect(screen.getByLabelText("Editable movie taste summary"))
    .toHaveValue("Summary: The viewer may enjoy epic fantasy adventures."));
  fireEvent.click(screen.getByText("Get Recommendations", { selector: "button" }));

  const results = (await screen.findByText("Recommended Movies", { selector: "h2" })).parentElement;
  const cards = within(results).getAllByRole("heading", { level: 3 });
  expect(cards.map((card) => card.textContent)).toEqual(reportedResponse.items.map((item) => item.title));
  expect(cards).toHaveLength(12);
  for (let rank = 1; rank <= 12; rank += 1) {
    expect(within(results).getByText(`#${rank}`)).toBeInTheDocument();
  }
  expect(mockPersistRender.mock.calls.at(-1)[1].map((item) => item.movie_id))
    .toEqual(reportedResponse.items.map((item) => item.movie_id));
  expect(axios.post.mock.calls.at(-1)[1].liked_movie_ids).toEqual([4993]);
  expect(within(results).queryByRole("heading", { name: /Fellowship of the Ring/ })).not.toBeInTheDocument();

  fireEvent.change(screen.getByRole("slider"), { target: { value: "2" } });
  fireEvent.click(screen.getByText("Get Recommendations", { selector: "button" }));
  const smaller = (await screen.findByText("Recommended Movies", { selector: "h2" })).parentElement;
  expect(within(smaller).getAllByRole("heading", { level: 3 }).map((card) => card.textContent))
    .toEqual(reportedResponse.items.slice(0, 2).map((item) => item.title));
  expect(mockPersistRender.mock.calls.at(-1)[1]).toHaveLength(2);
});
