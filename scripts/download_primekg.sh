#!/usr/bin/env bash
# Downloads PrimeKG (Chandak et al., Scientific Data 2023) from Harvard Dataverse (~1 GB).
set -euo pipefail
mkdir -p data
curl -L --fail -o data/kg.csv "https://dataverse.harvard.edu/api/access/datafile/6180620"
python -m rewind.data.kg build data/kg.csv data/primekg.pt
