# Site A — Electronic Health Records

Place EHR CSV files for **Site A** in this directory.

## Expected Files

| File | Description |
|------|-------------|
| `patients.csv` | Patient demographics — columns: `patient_id`, `age`, `gender`, `race` |
| `conditions.csv` | Diagnoses / conditions — columns: `patient_id`, `condition_code`, `condition_name`, `onset_date` |
| `labs.csv` | Lab results — columns: `patient_id`, `lab_name`, `lab_value`, `lab_unit`, `result_date` |
| `medications.csv` | Active medications — columns: `patient_id`, `medication_name`, `start_date`, `end_date` |

> The federated client will read whatever files are present.
> Missing files simply mean that category of criteria cannot be evaluated.
