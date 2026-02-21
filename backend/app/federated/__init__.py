"""
Federated Learning Module (Flower-based)

Simulates a federated patient-screening workflow using the
`Flower <https://flower.ai>`_ framework:

- A **Central Server** (Flower server with a custom strategy) broadcasts
  inclusion / exclusion criteria to participating clinical sites.
- Each **Federated Client** (Flower NumPyClient) reads its local EHR
  data, optionally uses MedGemma for criteria interpretation, evaluates
  patients, and reports *only* aggregate counts back.

Also includes federated treatment-arm monitoring components for
cross-site aggregate progress queries (adverse events, visits,
responses, dropouts, and lab trends).

No patient-level data ever leaves a site.
"""
