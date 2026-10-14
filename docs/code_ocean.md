# Code Ocean setup for Nature Communications

A capsule combines source code, input data, the software environment, and a reproducible execution entrypoint. GitHub tracks source changes; a published capsule can preserve an executable version with a DOI. Nature Communications offers Code Ocean as an optional route for code sharing and review. Code central to the paper must still be made available to editors/reviewers under the journal's requirements.

## Author steps

1. In the submission form, answer **Yes** to having developed code central to the work. If you want the offered Code Ocean support, select **Yes** for that separate optional service. Follow the journal's subsequent instructions; opting in is not the same as already having a published capsule.
2. Create a private capsule. Import an authorized GitHub repository or upload this package. Place the contents of `code/` in the capsule's code directory and the contents of `data/` in its data directory. Include the README, methods, and validation notes. Do not upload credentials or unrelated VPS files.
3. Set a Python 3.12 environment and install the package versions in `environment/requirements.lock.txt` during environment setup. `requirements.txt` separately lists the six primary scientific dependencies. The checked VPS interpreter was 3.12.3; local test interpreter details are in `docs/validation.md`. The Dockerfile is a portable environment recipe, not proof that Code Ocean has already built it.
4. Select `code/run` as the executable entrypoint. It runs both analyses using `../data` and writes results under `../results`. In the Code Ocean filesystem these resolve to `/data` and `/results`. Run a headless **Reproducible Run**, check completion, and inspect the generated CSVs, plots, and `run_manifest.json`. A terminal-only interactive run is not a substitute for this check.
5. Share the private reviewer link using the journal's instructions. Reviewers must be able to run the capsule without access to the author's private GitHub account. Preserve the final version and follow the platform/journal process for public release and DOI creation if the paper is accepted.

Do not insert a DOI or claim the capsule is available until it exists and has been tested. Review and preserve the existing repository license; this update does not select a new license on the author's behalf.

## Submission-form wording while preparing the capsule

> The custom Python code and input data underlying the machine-learning analyses will be provided to the editors and reviewers. We plan to share a private Code Ocean capsule containing the descriptor-model workflow, the retrospective sequential-search benchmark, pinned software dependencies, and a reproducible run script. The final code and data will be made publicly available upon publication, with an archived version linked in the Code availability statement.

Use this as a plan only if selecting that sharing route. Once the reviewer capsule is ready, replace future tense with its actual availability and provide the private reviewer link through the submission system. Do not expose a private review link in the public manuscript.

## Optional standalone Docker check

From the package root, on a machine with Docker:

```bash
docker build -t material-network-repro environment
mkdir -p results
docker run --rm -v "$PWD/code:/code:ro" -v "$PWD/data:/data:ro" -v "$PWD/results:/results" material-network-repro
```

This command recomputes both analyses. The scientific workflow does not need runtime network access. Docker execution was not inferred from local Python success; check `validation.md` for what was actually tested.

## Official references

- Nature Communications, [How to submit](https://www.nature.com/ncomms/submit/how-to-submit).
- Nature Communications, [Code sharing editorial](https://www.nature.com/articles/s41467-024-50517-4).
- Nature Portfolio, [Code sharing and Code Ocean integration](https://www.nature.com/articles/s41561-024-01523-0).
- Code Ocean, [Create a capsule](https://docs.codeocean.com/user-guide/v2.16.0/quick-start-guide/create-a-capsule-in-5-minutes).
- Code Ocean, [Reproducible Runs](https://docs.codeocean.com/user-guide/v4.0.0/compute-capsule-basics/reproducible-runs).
- Code Ocean, [GitHub integration](https://docs.codeocean.com/user-guide/v4.2.0/git-provider-integration-guide/github-integration-user-guide).

Policy and interface details were checked on 2026-10-14; follow any newer journal instructions shown in the live submission system.
