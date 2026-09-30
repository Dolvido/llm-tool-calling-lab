# LLM Tool Calling Lab Portfolio and Media Strategy

Revised for Luke Payne on September 29, 2026. This is a local strategy and copy draft for personal GitHub and the existing portfolio website. Implementation and evaluation are complete; use RESULTS.md, RELEASE_STATUS.md and the release assets for verified claims. Planned-stage wording below is historical draft copy, not current publication copy.

Lead with the useful system: an open-source, customizable chatbot that chooses appropriate ML approaches for a question, runs them, and uses recorded evidence in the conversation. Support that promise with an exercised extension example, reproducible comparisons, failures and clear limits.

## Project identity and audience

| Element | Current recommendation |
| --- | --- |
| Working public name | LLM Tool Calling Lab |
| Descriptive line | A customizable chatbot for choosing ML methods and inspecting their evidence |
| Proposed repository name | `llm-tool-calling-lab` |
| Proposed portfolio route | `/projects/llm-tool-calling-lab/` |
| Category | Applied AI systems and evaluation |
| Current status | Experimental v0.1.0 package; see RELEASE_STATUS.md and RESULTS.md |
| Primary audience | Developers exploring or building chatbots with specialist ML tools |
| Secondary audience | Engineers and hiring managers assessing Luke's implementation and experimental judgment |

Keep the existing personal identity, website and local project folder. Repository and route names are proposals, not created resources or claims of exclusive availability. No new branding campaign, company identity, domain or social channel is required.

The Outlier Tool Lab identity belonged to the earlier single-anomaly experiment. Its previous strategy is preserved in ARCHIVE_OUTLIER_MARKETING.md. Unusual-case ranking remains one example inside the broader prototype.

## The concept to communicate

A user should be able to describe what they want to learn without choosing an algorithm. The chatbot interprets the answer needed, checks which data is available, selects eligible approaches, runs a bounded comparison, and responds using those results. A useful follow-up can change the interpretation, candidate comparison, or next step.

The core sequence is:

Question -> answer needed -> suitable methods -> computation -> recorded evidence -> answer -> follow-up.

In version 0.1, adaptation means task/target interpretation, candidate selection, comparison and a supported next action. The first implementation uses fixed model architectures and small presets. Do not imply it invents ideal models, trains new foundation models, automatically solves causal questions or proves general reasoning improvements.

The initial implementation target is four numerical-table task families and two methods per family: numerical prediction, binary classification, unusual-case ranking and exploratory grouping. Report the actual retained and evaluated families at release. Generalizable here means reusable task/data contracts and replaceable components across numerical datasets.

The engineering value is giving other developers something they can run, inspect, modify and benchmark before investing in a larger system. Accuracy improvement is an outcome to investigate. A measured tie or loss can still accompany a useful reusable implementation.

## What to keep from the earlier strategy

Keep fair comparisons, held-out cases, visible failures, usage/cost and latency where measured, a verified local demo, and one concise case study. These make the broader concept credible.

Use two kinds of evidence:
- Operational evidence: a working conversation, appropriate question-to-method decisions, a consequential follow-up, a fresh installation and a successful adapter replacement.
- Empirical evidence: results against a generic LLM with the same tools and budget, plus deterministic model references.

Completion of the first does not prove superiority in the second. A single successful demo does not establish benchmark reliability.

The portfolio lesson should show what Luke made reusable, why the architecture was chosen, what failed, and what the measurements changed about the design. This builds on the emphasis on local AI tools and reproducible workflows in [Luke's GitHub profile](https://github.com/Dolvido/Dolvido/blob/main/README.md).

## GitHub presentation

The README's opening screen should answer: what does this do, what can I try, what is actually supported, and how can I replace a component?

After implementation, use this order:
1. A plain description, accurate release status and a short real demonstration.
2. A verified quickstart and supported-input/task table.
3. A compact architecture diagram showing where model results enter the conversation.
4. One exercised adapter replacement with the actual configuration change and output.
5. The measured comparison, including failures, task-specific quality, latency and available cost.
6. Reproduction instructions, frozen settings, limitations and extension directions.

Keep source code, configurations, model requirements, example data, dependency/license information and run evidence accessible. Separate replay mode from live model use. State the tested backend and the data that leaves the machine if a hosted backend is configured.

Evaluation should remain easy to find. Do not present different family metrics as one accuracy number or count repeat runs as independent datasets. A baseline win should be visible.

After a verified release, consider a selected-work link or pin. This strategy does not change the profile, create a repository or imply an actual pin.

## Portfolio story

Use one concise case study on the existing website. Its reader path is project card -> concrete demonstration -> architecture/customization -> evidence -> engineering decision -> GitHub.

The opening should explain what someone can do with the system. Then show how a question becomes a modeling approach and how returned evidence affects the answer. Use one real conversation, one extension example and the aggregate comparison rather than a long list of models.

Suggested case-study title:

A chatbot that chooses and uses specialist ML models

Suggested outline:
- The need: explore how specialist models can contribute to a conversation without requiring the user to choose algorithms.
- The build: a shared task contract, validated execution, explicit model requirements and inspectable results.
- The demonstration: a question, a useful method choice and a consequential follow-up.
- The customization: the exact adapter/configuration change and its working result.
- The evidence: where explicit planning helped, tied or failed against simpler alternatives.
- The decision: what Luke would keep, simplify or investigate next.

Use Luke's own explanation of the engineering decisions, edited against actual evidence. The available local strategy does not verify the current website's visual layout; apply its established presentation when a later publishing task inspects the site.

## Demonstration and visuals

The main visual should show the conversational workflow. A results chart is a separate supporting visual. This makes the system's purpose understandable while preserving the evidence behind performance claims.

Use one existing development example for an optional 60-90-second recording:
- 0-15 seconds: a person asks a concrete numerical question; the assistant clarifies the target if needed.
- 15-40 seconds: the eligible methods run and a compact evidence panel shows measured results.
- 40-65 seconds: a follow-up produces an actual comparison, evidence retrieval, clarification or justified limitation.
- 65-90 seconds: show the model catalog and the already verified adapter replacement or link to its short documentation.

Do not cram every task family into the clip. A separate support table shows breadth. Label the selected recording as an illustrative run; aggregate held-out results establish measured reliability. Keep the run ID and whether any pauses were edited. Never substitute invented scores or an unlabeled replay for live execution.

The adapter evidence can be a small before/after configuration view and linked run record; no extra demo app or second video is necessary. A verified conversation segment may be captured in week 2. Add the customization segment or link after the adapter exercise in build block 11 during week 3.

## Planned-stage copy

These drafts describe intent and can be reused when publication is separately undertaken. They must remain in planned-stage language until implementation and checks support stronger wording.

Repository description:

Planned open-source prototype of a chatbot that selects and compares specialist ML methods, uses recorded evidence in follow-up conversation, and exposes a documented model-extension interface.

Portfolio card:

**LLM Tool Calling Lab**  
A planned, customizable chatbot that turns questions into ML analyses and explains the evidence. The first prototype targets numerical prediction, classification, unusual-case ranking and grouping, with reusable model interfaces and a reproducible comparison.  
**Status:** Planned prototype

Short personal positioning:

I am designing an open-source reference for conversational ML tool use: how a chatbot chooses a modeling approach, how results influence the conversation, and how another developer can replace the models.

After release, rewrite these drafts around the actual supported functions and observed findings. Use planned, implemented, evaluated and reproducible as distinct evidence states.

## Media schedule and workload

The build remains 39 hours over three weeks. Presentation uses the existing three portfolio/writing hours per week; it adds no build hours. Technical docs and raw reports are build deliverables. Editorial work reuses them.

| Window in 2026 | Presentation work | Maximum hours | Evidence gate |
| --- | --- | ---: | --- |
| September 29 to October 4 | Refine positioning, planned copy and architecture storyboard | 3 | Current plan; no result claims |
| October 5 to October 11 | Outline the case study and capture the conversation segment when the run exists | 3 | Verified development conversation |
| October 12 to October 18 | Add customization proof after build block 11, edit against the report, and check links/assets | 3 | Exercised extension, actual support, frozen results and clean setup |

Missing evidence postpones the associated asset. Keep daily notes private. Publish at most one substantial post every two weeks. One case study, one optional clip and the repository presentation are sufficient for this release.

Distribution stays on personal GitHub and the existing portfolio website. No additional channels, paid promotion or analytics implementation are required. Stars, traffic and hiring responses are unknown and are not completion criteria.

## Claims and release evidence

| Claim | Evidence required before using it |
| --- | --- |
| Working chatbot | A verified live question/tool/answer exchange and supported follow-up |
| Customizable | An exercised adapter/configuration replacement with a recorded result |
| Supports four task families | Each advertised family actually implemented and its status disclosed |
| Reproducible | Fresh installation plus documented data/configuration and rerunnable evaluation |
| Open source | Source and license actually made publicly available |
| Improves answers or selection | A specific measured comparison with denominators, failures and scope |
| Makes the user need less expertise | A separate appropriate user study; not established by this prototype |

Do not use novel cognition, ideal model for any question, predicts the future, production-ready or equivalent broad claims. The intended benefit is reducing uncertainty about building and extending this architecture.

## Current sources and upkeep

PLAN.md controls prototype scope and acceptance. EVALUATION.md controls comparisons and score interpretation. SCHEDULE.md controls the 39-hour build allocation and presentation ceiling. README.md is the entry point. This strategy translates that evidence into public presentation; it does not create implementation progress.

This revision draws on this project's concept/planning conversation and the reviewed chat **Name and market portfolio project**. It keeps that chat's evidence, personal-channel and workload discipline while replacing its superseded anomaly-only positioning.

[Luke's GitHub profile](https://github.com/Dolvido/Dolvido/blob/main/README.md) was read on September 29, 2026. The portfolio homepage could not be retrieved through the web tool in this review; no current visual audit or live website changes are claimed.

The referenced chat has a separate ongoing documentation-automation request. This revision neither configures nor changes its automations and does not send it a message. Any later documentation refresh should use the current local plan and verified evidence rather than archived scope. No publication is performed here.


