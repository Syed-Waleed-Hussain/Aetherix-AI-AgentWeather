# Deploy the historical dashboard at PKR 0

Streamlit Community Cloud offers free hosting. This dashboard needs a GitHub connection, Python 3.12, the repository's dashboard dependencies, and the bundled 93 KB forecast snapshot. It uses no billing setup or cloud service credentials. [Community Cloud overview](https://docs.streamlit.io/deploy/streamlit-community-cloud)

## Community Cloud

1. After the owner approves release, commit and push the reviewed Day 8 changes to `main` in `Syed-Waleed-Hussain/Aetherix-AI-AgentWeather`.
2. Sign in with GitHub and choose **Create app → Yup, I have an app**. [Community Cloud workspace](https://share.streamlit.io/)
3. Select repository `Syed-Waleed-Hussain/Aetherix-AI-AgentWeather`, branch `main`, and main file path `dashboard/app.py`.
4. In **Advanced settings**, select **Python 3.12**, leave **Secrets** blank, and click **Save**. Let Community Cloud assign a subdomain unless you have chosen one.
5. Click **Deploy** after public deployment is authorized. Check the app logs. Open the assigned address, check that the header reads **HISTORICAL** and **Live observation · UNAVAILABLE**, and navigate all six sections. [Deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## Configuration and constraints

| Setting | Hosted snapshot | Local workspace |
| --- | --- | --- |
| Entry point | `dashboard/app.py` | Same |
| Python dependencies | `dashboard/requirements.txt` beside the entry point | Dashboard requirements plus local model dependencies for inference |
| `AETHERIX_MODE` | Clean checkouts automatically select snapshot mode | Set to `snapshot` to preview hosted behavior |
| `AETHERIX_SNAPSHOT_PATH` | Unset to use the bundled JSON | Optional path to a trusted local saved state |
| `AETHERIX_AGENT_DB` | Unused | Optional SQLite path, defaults to `outputs/agents/decisions.sqlite3` |
| Groq or Hopsworks secrets | Leave blank | Optional, for explicitly invoked integrations |

The Community Cloud dependency resolver looks beside the entry point before the repository root. The dashboard uses one cached bundled document and no training data or checkpoint loads in snapshot mode. Community Cloud apps can sleep or hit resource limits. Verify startup after the first deployment. [Dependency rules](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies) · [Resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)

Snapshot mode includes the saved issue-hour inputs, 24-hour forecasts, alert precautions by audience, deterministic report, matched scores, and a drift summary. The full SQLite decision history, local inference, detailed drift measurements, and Hopsworks probe are available in the local workspace. Current data remains unavailable until a successfully fetched, timestamped current observation is actually connected. The existing hourly check reads provider-modeled hourly estimates and has no live inference bridge.

## Local preview

From the project root with the Python 3.12 environment already installed:

```powershell
$env:AETHERIX_MODE = 'snapshot'
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.address 127.0.0.1
```

To restore local mode, unset `AETHERIX_MODE`. The local forecast button requires the feature dataset, fitted estimators with their original scikit-learn version, and LangGraph. The default report uses the deterministic template. An existing free-tier Groq key and model name can be configured locally with `GROQ_API_KEY` and `GROQ_MODEL` for the optional layout. Keep keys in local ignored configuration or Streamlit secrets.
