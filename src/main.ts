import { readFileSync } from "fs";
import * as core from "@actions/core";
import OpenAI from "openai";
import { Octokit } from "@octokit/rest";
import parseDiff, { Chunk, File } from "parse-diff";
import minimatch from "minimatch";
import { ChatCompletionCreateParamsNonStreaming } from "openai/resources";

const GITHUB_TOKEN: string = core.getInput("GITHUB_TOKEN");
const OPENAI_API_KEY: string = core.getInput("OPENAI_API_KEY");
const OPENAI_API_MODEL: string = core.getInput("OPENAI_API_MODEL");

const octokit = new Octokit({ auth: GITHUB_TOKEN });

const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
});

interface PRDetails {
  owner: string;
  repo: string;
  pull_number: number;
  title: string;
  description: string;
}

async function getPRDetails(): Promise<PRDetails> {
  const { repository, number } = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH || "", "utf8")
  );
  const prResponse = await octokit.pulls.get({
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
  });
  return {
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
    title: prResponse.data.title ?? "",
    description: prResponse.data.body ?? "",
  };
}

async function getDiff(
  owner: string,
  repo: string,
  pull_number: number
): Promise<string | null> {
  const response = await octokit.pulls.get({
    owner,
    repo,
    pull_number,
    mediaType: { format: "diff" },
  });
  // @ts-expect-error - response.data is a string
  const diff = response.data;
  if (!diff.trim()) {
    return null;
  }
  return diff;
}

async function analyzeCode(
  parsedDiff: File[],
  prDetails: PRDetails
): Promise<Array<{ body: string; path: string; line: number }>> {
  const comments: Array<{ body: string; path: string; line: number }> = [];

  for (const file of parsedDiff) {
    if (file.to === "/dev/null" || file.chunks.length === 0) continue; // Ignore deleted and blank files
    for (const chunk of file.chunks) {
      const prompt = createPrompt(file, chunk, prDetails);
      const aiResponse = await getAIResponse(prompt);
      if (aiResponse) {
        const newComments = createComment(file, chunk, aiResponse);
        if (newComments) {
          comments.push(...newComments);
        }
      }
    }
  }
  return comments;
}

function createPrompt(file: File, chunk: Chunk, prDetails: PRDetails): string {
  const botName = core.getInput("bot_name");
  const rules = core.getInput("rules");
  const rulesPrompt =
    rules === ""
      ? ""
      : `Your review will *only* ensure the following rules are followed:
${rules}`;
  return `Your name is ${botName}. Your task is to review pull requests. ${rulesPrompt}

Here are your instructions regarding the format and the style of the review:
- Provide the response in the following JSON format: {"reviews": [{"lineNumber": <line_number>, "reviewComment": "<review comment>"}]}
- Provide comments and suggestions ONLY if there is something to improve regarding code style or potential errors, otherwise "reviews" should be an empty array.
- Write the comment in GitHub Markdown format.
- Suggest a fix in a reviewComment if applicable by using a \`\`\`suggestion\`\`\` code block (with proper whitespace indentation).
- Use the given description only for overall context and focus only on the code.
- IMPORTANT: NEVER suggest adding comments to the code.
- Do not review commented sections of code or already corrected errors in the code. Review only the latest updates.
- Leave comments only on things that could potentially cause an error in the code or that do not match the code style.

Review the following code diff in the file "${file.to}" and take the pull request title and description into account when writing the response.

Pull request title: ${prDetails.title}
Pull request description:

---
${prDetails.description}
---

Git diff to review:

\`\`\`diff
${chunk.content}
${chunk.changes
  // @ts-expect-error - ln and ln2 exists where needed
  .map((c) => `${c.ln ? c.ln : c.ln2} ${c.content}`)
  .join("\n")}
\`\`\`
`;
}

/**
 * Checks if the given model supports JSON object mode.
 *
 * The function determines if a model supports JSON object mode by checking
 * if the model's name includes any of the prefixes specified in the
 * `supportedJsonObjectModelsPrefix` array. The supported models include
 * variants of GPT-4-turbo, GPT-3.5-turbo, and GPT-4o.
 *
 * @param {ChatCompletionCreateParamsNonStreaming["model"]} model - The name of the model to check.
 * @returns {boolean} - Returns true if the model supports JSON object mode, false otherwise.
 */
const isJsonObjectSupportedModel = (
  model: ChatCompletionCreateParamsNonStreaming["model"]
): boolean => {
  // Models supported by JsonObject, reference https://platform.openai.com/docs/guides/text-generation/json-mode
  const supportedJsonObjectModelsPrefix = [
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "gpt-4o",
  ];

  return supportedJsonObjectModelsPrefix.some((item) => model.includes(item));
};

async function getAIResponse(prompt: string): Promise<Array<{
  lineNumber: string;
  reviewComment: string;
}> | null> {
  const queryConfig = {
    model: OPENAI_API_MODEL,
    temperature: 0.2,
    max_tokens: 1000,
    top_p: 1,
    frequency_penalty: 0,
    presence_penalty: 0,
    ...(isJsonObjectSupportedModel(OPENAI_API_MODEL)
      ? { response_format: { type: "json_object" } }
      : {}),
  };

  try {
    const response = await openai.chat.completions.create({
      ...(queryConfig as ChatCompletionCreateParamsNonStreaming),
      messages: [
        {
          role: "system",
          content: prompt,
        },
      ],
    });

    const rawContent = response.choices[0].message?.content || "";

    const startIndex = rawContent.indexOf("{");
    const endIndex = rawContent.lastIndexOf("}") + 1;

    if (startIndex === -1 || endIndex === -1) {
      throw new Error("No valid JSON found in response content.");
    }

    const jsonString = rawContent.substring(startIndex, endIndex).trim();

    return JSON.parse(jsonString).reviews;
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
}

function createComment(
  file: File,
  chunk: Chunk,
  aiResponses: Array<{
    lineNumber: string;
    reviewComment: string;
  }>
): Array<{ body: string; path: string; line: number }> {
  return aiResponses.flatMap((aiResponse) => {
    if (!file.to) {
      return [];
    }
    console.log(`Creating comment for file: ${file.to}, line: ${aiResponse.lineNumber}`);
    return {
      body: aiResponse.reviewComment,
      path: file.to,
      line: Number(aiResponse.lineNumber),
    };
  });
}

async function createReviewComment(
  owner: string,
  repo: string,
  pull_number: number,
  comments: Array<{ body: string; path: string; line: number }>
): Promise<void> {

  console.log("Review comments to be posted:", comments);

  const validComments = comments.filter(comment => comment.path && comment.line > 0);

  if (validComments.length === 0) {
    console.log("No valid comments to post");
    return;
  }

  await octokit.pulls.createReview({
    owner,
    repo,
    pull_number,
    comments: validComments,
    event: "COMMENT",
  });
}

async function main() {
  const prDetails = await getPRDetails();
  let diff: string | null;
  const eventData = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH ?? "", "utf8")
  );

  if (eventData.action === "opened") {
    diff = await getDiff(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number
    );
  } else if (eventData.action === "synchronize") {
    const newBaseSha = eventData.before;
    const newHeadSha = eventData.after;

    const response = await octokit.repos.compareCommits({
      headers: {
        accept: "application/vnd.github.v3.diff",
      },
      owner: prDetails.owner,
      repo: prDetails.repo,
      base: newBaseSha,
      head: newHeadSha,
    });

    diff = String(response.data);
    if (!diff.trim()) {
      diff = null;
    }
  } else {
    console.log("Unsupported event:", process.env.GITHUB_EVENT_NAME);
    return;
  }

  if (!diff) {
    console.log("No diff found or diff is empty");
    return;
  }

  const parsedDiff = parseDiff(diff);

  const excludePatterns = core
    .getInput("exclude")
    .split(",")
    .map((s) => s.trim());

  const filteredDiff = parsedDiff.filter((file) => {
    return !excludePatterns.some((pattern) =>
      minimatch(file.to ?? "", pattern)
    );
  });

  const comments = await analyzeCode(filteredDiff, prDetails);
  if (comments.length > 0) {
    await createReviewComment(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number,
      comments
    );
  }
}

main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
