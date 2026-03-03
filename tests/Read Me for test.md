# Read Me for Test

This document describes installation, testing, and deployment for the KNIME Google Earth Engine Extension test workflows.

---

## GEE Authenticator

All workflows that use Google Earth Engine nodes require authentication. The **Google Earth Engine Connector** node uses the **Google Authenticator** node (from KNIME’s integration) to establish the connection. You can use either interactive authentication (personal Google account) or a service account JSON key.

### Account Registration

1. **Sign up for Google Earth Engine** — Visit [signup.earthengine.google.com](https://signup.earthengine.google.com/) and register with your Google account. Approval may take a short time.

2. **Link a Google Cloud Project** — As of 2024, Earth Engine accounts must be linked to a Google Cloud Project:
   - Visit [Earth Engine Code Editor](https://code.earthengine.google.com/)
   - Click your avatar (top-right) → **Register a new Cloud Project** (or use an existing project)

3. **Project ID** — You need the **Project ID** for the Google Earth Engine Connector node. Find it in the Code Editor by clicking your avatar, or in [Google Cloud Console](https://console.cloud.google.com/) under your project.

### Interactive Authentication (Personal Account)

- In the **Google Authenticator** node, select **Interactive** authentication
- Set scope to **Custom** and add: `https://www.googleapis.com/auth/earthengine`
- For Drive exports, also add: `https://www.googleapis.com/auth/drive`
- For Cloud Storage exports, also add: `https://www.googleapis.com/auth/cloud-platform`
- Sign in with your Google account that has Earth Engine access

### Service Account (JSON Key)

For automated or non-interactive workflows:

1. **Create a service account**
   - Go to [Google Cloud Console → IAM & Admin → Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts/)
   - Click **+ CREATE SERVICE ACCOUNT**
   - Name it and create it within your Earth Engine–enabled project

2. **Create a JSON key**
   - Open the service account → **Keys** tab → **Add Key** → **Create new key**
   - Choose **JSON** and download the file
   - Store the file securely; do not share or commit it to version control

3. **Register the service account for Earth Engine**
   - The Google Cloud project must be registered for Earth Engine
   - Add the **Service Usage Consumer** role to the service account:
     - IAM & Admin → IAM → find the service account → Edit
     - Add role **Service Usage Consumer** → Save

4. **Use in KNIME**
   - In the **Google Authenticator** node, select **Google Service Account JSON Key**
   - Upload the JSON key file
   - Set scope to **Custom** and add: `https://www.googleapis.com/auth/earthengine`
   - For Cloud exports, add: `https://www.googleapis.com/auth/cloud-platform`
   - **Note:** Drive export is not supported for service accounts

---

## Installation

### Prerequisites

The following platform and extensions must be installed before using this extension and its test workflows:

1. **KNIME Analytics Platform** — Download and install from [knime.com](https://www.knime.com/downloads).

2. **KNIME Geospatial Analytics Extension** — Install via KNIME:
   - Open KNIME Analytics Platform
   - Go to **File → Install KNIME Extensions**
   - Search for `geospatial` or `geo` and install **KNIME Geospatial Extension**
   - This provides geometry data types and geospatial capabilities required by the workflows

3. **KNIME Google Earth Engine Extension** — Install from [KNIME Hub](https://hub.knime.com/) once available, or follow the [Contribution Guide](../CONTRIBUTING.md#setup) for local development setup.

---

## Testing

The `tests` folder contains workflows and sample data that can be run directly to verify the extension.

### Test Workflows

---

#### Workflow 1: G2SFCA with Heat Exposure

The G2SFCA workflow (`KNIME-GEE-G2SFCA-RRE.knwf`) demonstrates an accessibility study extended with heat exposure analysis. It combines:

- **Accessibility (G2SFCA)** — A gravity-based Two-Step Floating Catchment Area (2SFCA) model for measuring spatial accessibility to services (e.g., hospitals).
- **Heat Exposure** — Land surface temperature derived from Landsat ST_B10 thermal band, showing the integration of geospatial analytics with Google Earth Engine.

**Input Data**

- **Baton Rouge census block groups** — Demand-side population data (`BR_Bkg.zip`).
- **Hospital data** — Supply-side facility locations and capacities (`hosp.geojson`), from the accessibility case in *Computational Methods and GIS Applications in Social Science*, Chapter 2.

**Workflow Components**

1. **Travel cost** — Uses the **OSRM** (Open Source Routing Machine) node to compute travel OD (origin–destination) matrices for accessibility modeling.
2. **Accessibility components** — Includes decay functions, 2SFCA score calculation, and related accessibility model nodes.
3. **Land surface temperature** — Uses Google Earth Engine to:
   - Read Landsat image collection (e.g., `LANDSAT/LC09/C02/T1_L2`)
   - Filter by cloud cover and spatial extent
   - Aggregate (e.g., mean) and extract the ST_B10 band
   - Compute zonal statistics with the **Feature Collection Reducer** node

This workflow illustrates how the Geospatial Analytics Extension and Google Earth Engine can be used together for accessibility and environmental analysis.

---

#### Workflow 2: Chapter 6 Image Classification and Clustering (Textbook Reproduction)

The image classification workflow (`KNIME-GEE--Image-Classification.knwf`) reproduces **Chapter 6: Image Manipulation—Classification/Clustering** from the open-source textbook *Cloud-Based Remote Sensing with Google Earth Engine: Fundamentals and Applications* (2024).

The workflow demonstrates supervised classification (CART, Random Forest) and unsupervised clustering (K-means) using Landsat imagery. **GEE Image View** outputs align with the textbook figure numbers:

| GEE Image View Node        | Figure in Book  |
|---------------------------|-----------------|
| Landsat Image             | Fig. 6.2        |
| CART classification       | Fig. 6.11       |
| Random forest classified  | Fig. 6.13       |
| K-means classification    | Fig. 6.15       |

**Workflow Sections**

1. **Read Image** — Image Collection Reader (`LANDSAT/LC08/C02/T1_L2`), spatial and general filters (date, cloud cover), Image Collection Aggregator.
2. **Read Labeled Points** — CSV Reader for training labels, GeoJSON to Geometry, Column Filter.
3. **Local to Cloud** — GeoTable to Feature Collection for ROI and labeled points.
4. **Bands Sampling** — Reference Feature Collection Sampler to extract band values at labeled points.
5. **Supervised Classification** — Feature Collection Partitioner, GEE CART Learner, GEE Random Forest Learner, Image Class Predictor.
6. **Unsupervised Clustering** — Image Random Sampling, K-means Clustering Learner, Apply Clustering.
7. **Visualization** — GEE Image View nodes for each result.

This workflow shows how to replicate the textbook examples in KNIME using the GEE extension for remote sensing classification and clustering.

---

## Deployment

For production or shared use:

- **KNIME Hub** — Upload workflows to [KNIME Hub](https://hub.knime.com/) for sharing and version control.
- **Extension** — Install the KNIME Google Earth Engine Extension from the KNIME Hub when published.
- **Requirements** — Ensure all prerequisite extensions and a valid Google Earth Engine account (for GEE nodes) are available in the deployment environment.

---

## More Examples

For additional workflows and tutorials, see:

**[Google Earth Engine Examples — Center for Geographic Analysis at Harvard University](https://hub.knime.com/center%20for%20geographic%20analysis%20at%20harvard%20university/spaces/Google%20Earth%20Engine%20Examples/~eheY8uN9VYNQ8oZq/)**
