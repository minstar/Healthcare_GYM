# Medical MCP Docker Environment

Self-contained, stateless Docker environment serving 11 medical MCP servers
behind a small HTTP API for medical trajectory synthesis and evaluation.

## Origin

The MCP server implementations use the same packages as MCP-Atlas evaluation to
prevent tool hallucination (schema mismatch between training and eval).

- **clinicaltrialsgov-mcp-server**: `npm clinicaltrialsgov-mcp-server@1.0.8`
- **pubmed**: `git+https://github.com/geobio/PubMed-MCP-Server.git` via uvx
- **openfda**: `npm bach-openfda` ([BACH-AI-Tools/mcp-openfda](https://github.com/BACH-AI-Tools/mcp-openfda)) — drug labels, adverse reactions, warnings, indications (openFDA API, no key required)
- **opentargets**: `git+https://github.com/opentargets/open-targets-platform-mcp` via uvx — drug-target-disease associations, GraphQL queries (public API, no key required)
- **chembl**: [Augmented-Nature/ChEMBL-MCP-Server](https://github.com/Augmented-Nature/ChEMBL-MCP-Server) — compounds, targets, bioactivity, drug development, ADMET (EBI ChEMBL REST API, no key required)
- **uniprot**: [Augmented-Nature/UniProt-MCP-Server](https://github.com/Augmented-Nature/Augmented-Nature-UniProt-MCP-Server) — protein search, sequences, structure, domains, pathways (UniProt REST API, no key required)
- **pubchem**: [Augmented-Nature/PubChem-MCP-Server](https://github.com/Augmented-Nature/PubChem-MCP-Server) — compound search, molecular properties, bioassays, safety data (NIH PubChem API, no key required)
- **kegg**: [Augmented-Nature/KEGG-MCP-Server](https://github.com/Augmented-Nature/KEGG-MCP-Server) — pathways, genes, compounds, reactions, enzymes, diseases, drugs (KEGG REST API, no key required)
- **ncbi-datasets**: [Augmented-Nature/NCBI-Datasets-MCP-Server](https://github.com/Augmented-Nature/NCBI-Datasets-MCP-Server) — genomes, genes, taxonomy, assemblies, proteins (NCBI Datasets v2 API, key optional)
- **healthcare**: `npm healthcare-mcp` ([Cicatriiz/healthcare-mcp-public](https://github.com/Cicatriiz/healthcare-mcp-public)) — FDA, ICD-10, medRxiv, NCBI Bookshelf, DICOM, BMI calculator (public APIs, no key required)
- **biomcp**: `pip biomcp-cli` ([genomoncology/biomcp](https://github.com/genomoncology/biomcp)) — genes, variants, articles, trials, drugs, diseases, pathways, proteins, adverse events, PGx (40+ upstream APIs, most free)

## Build

```bash
./build.sh medical-mcp-env:1.1   # builds image + saves tar to images/
```

## Extract Tool Specs

After building, extract actual tool specs from the running container:

```bash
./extract_tool_specs.sh 6986 tool_specs_medical.json
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/list-tools` | POST | List all available MCP tools |
| `/call-tool` | POST | Call a tool with `{tool_name, tool_args, use_cache}` |
| `/reset-state` | POST | Clear response cache |
| `/cache-stats` | GET | Cache size and TTL info |

## Notes

- All servers call **live public APIs** — no local data bundled, internet required
- Response cache (48h TTL) reduces redundant API calls during synthesis
- No API keys or subscriptions needed
