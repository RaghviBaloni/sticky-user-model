# Self red-team

**Stage 5.** Write the three strongest objections to your own result, then address or concede each.
Conceding well is worth more than a weak rebuttal.

---

## Objection 1

- **The objection:**
- **Is it right?**
- **Evidence against it (or concession):**
- **Goes in Limitations?** ☐

## Objection 2

## Objection 3

---

## Standing objections to check every time

- [ ] Could a bag-of-words / sentiment baseline explain this?
- [ ] Is the final user turn byte-identical across the compared conditions?
- [ ] Were any hooks still attached during measurement?
- [ ] Was a generation-time KV cache reused for measurement?
- [ ] Does the position control produce a similar effect?
- [ ] Does the random-direction control produce a similar effect?
- [ ] Is length a confound?
- [ ] Is the generated text still coherent at this steering strength?
- [ ] Probe accuracy suspiciously high (>97%)?
- [ ] How many seeds is this based on?
- [ ] Would this survive on a second model?
