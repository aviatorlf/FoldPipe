# Contributing to FoldPipe

First off, thank you for considering contributing to FoldPipe! It's people like you that make FoldPipe such a great tool for the computational biology community.

## 1. Where do I go from here?

If you've noticed a bug or have a feature request, make one! It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

## 2. Fork & create a branch

If this is something you think you can fix, then fork FoldPipe and create a branch with a descriptive name.

A good branch name would be (where issue #325 is the ticket you're working on):

```sh
git checkout -b 325-add-new-dataloader
```

## 3. Implementation Guidelines

- **Keep Data and Logic Separate:** The core loaders in `foldpipe/` should never hardcode paths.
- **Maintain High Performance:** Ensure that any modifications to the dataloader do not introduce CPU bottlenecks. Always test with `pin_memory=True`.
- **Test:** Run the provided benchmark script (`scripts/run_benchmark.py`) to verify no performance regressions are introduced.

## 4. Make a Pull Request

At this point, you should switch back to your master branch and make sure it's up to date with FoldPipe's master branch:

```sh
git remote add upstream git@github.com:aviatorlf/FoldPipe.git
git checkout main
git pull upstream main
```

Then update your feature branch from your local copy of master, and push it!

```sh
git checkout 325-add-new-dataloader
git rebase main
git push --set-upstream origin 325-add-new-dataloader
```

Finally, go to GitHub and make a Pull Request.
