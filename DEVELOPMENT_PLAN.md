# HeadSwap implementation status

- Scope: standalone Russian Windows desktop app; local neural whole-head swap.
- Modes: mutual exchange on one photo; donor portrait to a selected head.
- Selection: one/two head rectangles; generation is automatic inside those regions.
- Backend: FLUX.2 Klein 4B NF4, BFS Head V1 4B LoRA, FP16 compute, FP32 VAE,
  sequential loading and model CPU offload for RTX 2060 6 GB.
- Runtime is offline; only the installer downloads dependencies/model weights.
- Installers download large files in resumable 8 MiB chunks and verify SHA-256.
- Unit verification: 6 imaging tests passed (outside-region pixel identity,
  mutual compositing, invalid/overlapping selections, inference sizes,
  opaque replacement at the image boundary, flat-background color matching).
- Desktop window inspected visually; computer use was stopped with Esc.
- Installation complete. Dependency compatibility check passed for 48 packages.
- Real offline GPU inference verified: portrait transfer 41.2 seconds; mutual
  exchange with the final prompt 41.6 seconds, peak CUDA allocation 3776 MiB.
  Final output inspected visually; outside-selection pixel identity verified.
- Illustrated-character exchange verified: 120.4 seconds. Final background
  matching inspected visually and outside-selection pixels checked unchanged.
- Implementation and local QA complete. This is a standalone repository.