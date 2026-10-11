# Phase 4C source identity and population limits

This supplements the existing audit without changing the preserved data or
historical results. It does not assert that the source is a complete physical
vehicle census.

A read-only scan of the previously examined 6,078 clipped historical pilot JSON
records found median record duration **8.14 s**, interquartile range **3.68–16.55 s**;
**19.61%** of records span at least 20 s. These are durations after the old regional
extraction, not complete physical vehicle lifetimes. The original JSON's source
SHA256 is recorded in `outputs/i24_phase4c/source_identity_diagnostic.json`.

Both `fragment_ids` and `merged_ids` have median and 95th percentile length one,
maximum two. No fragment ID repeats between these selected records. This confirms
preserved record identifiers; it does not independently verify physical identity
correspondence across short source records.

There are 5,873 `Overlap`, 1,438 `Lost`, and 51 `Anomalous state` flag occurrences.
Do **not** interpret `Overlap` as a collision annotation or proof of duplicate
vehicles. The publisher describes these flags only as internal tracking-process
metadata. They cannot be turned into a new exclusion rule without validated
semantics, and no records were excluded based on this inspection.

The publisher's [v1.x documentation](https://github.com/I24-MOTION/I24M_documentation/blob/9f8a9158ba0bcbfdd525f4be771f8d45ad1bd4df/v1.x_I-24MOTION_data_documentation.pdf)
defines `merged_ids` as combined temporally overlapping fragments and
`fragment_ids` as combined non-overlapping fragments. It also documents missing
coverage, occlusions, partial post-processing recovery, and homography artifacts
that can create multiple trajectories for one vehicle. That documents a possible
failure mechanism; it does not quantify its incidence in this selected region.
The source repository was read at commit
`9f8a9158ba0bcbfdd525f4be771f8d45ad1bd4df`.

Consequences for the experiment:

- A disappearing record need not mean a physical roadway exit; a new identity need
  not mean a physical entrant. Population residuals mix true turnover, tracking
  loss/reacquisition, and possible fragmentation.
- Twenty-second micro scores condition on the recorded portion of the original
  observed cohort that still has usable target samples. Report endpoint counts
  and do not describe this survivor-conditioned score as all-vehicle accuracy.
- Compare generated overlap rates with recorded overlap diagnostics. Bounding-box
  overlap is a geometric proxy, not a verified crash event. Pair fractions and
  fractions of vehicles involved in overlaps have different denominators.
- Cohort density/flow and residual density/flow are additive label partitions;
  speed is not additive. A total-field decoder cannot be forced to equal the
  surviving historical cohort's generated fields.
- The joint objective is population-aware supervision of two distinct targets.
  It does not solve source identity reconciliation or generate individual future
  arrivals. Those remain required before an open-population conservation claim.

Citation: Gloudemans et al. (2023), *I-24 MOTION: An instrument for freeway traffic
science*, Transportation Research Part C 155, 104311.
