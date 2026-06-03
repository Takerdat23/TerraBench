import json
import os
from pathlib import Path
from datetime import datetime
import calendar
import subprocess

import certifi
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import cdsapi

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

from dotenv import load_dotenv

load_dotenv()

KEY = os.getenv("CDS_API_KEY")
BASE_DIR = Path(__file__).resolve().parent.parent
_COMBINED_BUNDLE_PATH = Path("/tmp/crawl_combined.pem")
_CA_ENV_KEYS: tuple[str, ...] = ("CDS_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE")
_VERIFY_FLAG_ENV = "DATA_CRAWL_VERIFY"

# Allowed CMIP6 variable names (UI labels) for validation.
_CMIP6_ALLOWED_VARIABLES = {
    "air temperature",
    "daily maximum near-surface air temperature",
    "eastward near-surface wind",
    "evaporation including sublimation and transpiration",
    "grid-cell area for atmospheric grid variables",
    "land ice area percentage",
    "near-surface air temperature",
    "near-surface specific humidity",
    "northward near-surface wind",
    "percentage of the grid cell occupied by land including lakes",
    "relative humidity",
    "sea floor depth below geoid",
    "sea level pressure",
    "sea surface salinity",
    "sea-ice area percentage on ocean grid",
    "snow depth",
    "specific humidity",
    "surface altitude",
    "surface downward northward wind stress",
    "surface downwelling shortwave radiation",
    "surface temperature",
    "surface upward latent heat flux",
    "surface upwelling longwave radiation",
    "toa incident shortwave radiation",
    "toa outgoing shortwave radiation",
    "total runoff",
    "capacity of soil to store water",
    "daily minimum near-surface air temperature",
    "eastward wind",
    "geopotential height",
    "grid-cell area for ocean variables",
    "moisture in upper portion of soil column",
    "near-surface relative humidity",
    "near-surface wind speed",
    "northward wind",
    "precipitation",
    "sea area percentage",
    "sea ice thickness",
    "sea surface height above geoid",
    "sea surface temperature",
    "sea-ice mass per area",
    "snowfall flux",
    "surface air pressure",
    "surface downward eastward wind stress",
    "surface downwelling longwave radiation",
    "surface snow amount",
    "surface temperature of sea ice",
    "surface upward sensible heat flux",
    "surface upwelling shortwave radiation",
    "toa outgoing longwave radiation",
    "total cloud cover percentage",
}

def _normalize_var_name(name: str) -> str:
    return name.strip().lower().replace("_", " ")

def _ok(payload):
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload))], metadata=payload)


def _error(msg: str) -> ToolResponse:
    return ToolResponse(
        content=[TextBlock(type="text", text=f"Error: {msg}")],
        metadata={"error": True, "message": msg},
    )


def _resolve_verify_setting():
    flag = os.getenv(_VERIFY_FLAG_ENV)
    if isinstance(flag, str) and flag.lower() in {"0", "false", "no", "off"}:
        return False

    env_values: list[str] = []
    ca_paths: list[Path] = []
    for env_key in _CA_ENV_KEYS:
        value = os.getenv(env_key)
        if not value:
            continue
        env_values.append(value)
        expanded = Path(value).expanduser()
        if not expanded.is_absolute():
            candidate = BASE_DIR / expanded
            if candidate.exists():
                expanded = candidate
        if expanded.exists():
            ca_paths.append(expanded)

    if ca_paths:
        try:
            combined_parts = [Path(certifi.where()).read_text()]
            for path in ca_paths:
                try:
                    combined_parts.append(path.read_text())
                except FileNotFoundError:
                    continue
            _COMBINED_BUNDLE_PATH.write_text("\n".join(combined_parts))
            return str(_COMBINED_BUNDLE_PATH)
        except Exception:
            return str(ca_paths[0])

    if env_values:
        return env_values[0]

    return True


def _resolve_key(override: str | None) -> str:
    key = override or KEY or os.getenv("CDSAPI_KEY")
    if not key:
        raise ValueError("CDS API key not provided. Set CDS_API_KEY or pass KEY parameter.")
    return key


def _validate_with_datastore(collection_id: str, request: dict, keys: list[str]) -> tuple[bool | None, str]:
    """
    Optionally validate requested values against CDS availability using ecmwf.datastores.

    Returns (status, note):
        status: True if valid, False if invalid, None if validation was skipped.
    """
    try:
        from ecmwf.datastores import Client as DSClient  # type: ignore
    except Exception as exc:
        return None, f"Constraint validation skipped (ecmwf.datastores missing): {exc}"

    try:
        client = DSClient()
        constraints = client.apply_constraints(collection_id, {k: request[k] for k in keys if k in request})
    except Exception as exc:
        return None, f"Constraint validation skipped (apply_constraints failed): {exc}"

    for key in keys:
        if key not in request or key not in constraints:
            continue
        req_vals = request[key]
        if not isinstance(req_vals, (list, tuple)):
            req_vals = [req_vals]
        req_vals = [str(v) for v in req_vals]
        valid_vals = constraints[key]
        if isinstance(valid_vals, (list, tuple)):
            valid_vals_str = [str(v) for v in valid_vals]
            missing = [v for v in req_vals if v not in valid_vals_str]
            if missing:
                return False, f"{key} values not available: {missing}; valid examples: {valid_vals_str[:10]}"
    return True, "Constraint validation passed via apply_constraints."


def _constraints_for_request(collection_id: str, request: dict) -> tuple[dict | None, str]:
    """Return constraint dictionary for a partial request, or (None, note) if unavailable."""
    try:
        from ecmwf.datastores import Client as DSClient  # type: ignore
    except Exception as exc:
        return None, f"Constraint lookup skipped (ecmwf.datastores missing): {exc}"

    try:
        client = DSClient()
        constraints = client.apply_constraints(collection_id, request)
        return constraints, "Constraint lookup succeeded."
    except Exception as exc:
        return None, f"Constraint lookup failed: {exc}"
 
def _resolve_meteoaq_save_path(
    save_path: str | os.PathLike[str],
    *,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> Path:
    """Resolve a MeteoAQ CSV path from either a directory or a filepath."""
    target = Path(save_path).expanduser()
    default_name = f"meteoaq_{lat:.4f}_{lon:.4f}_{start_date}_{end_date}.csv"

    if target.exists() and target.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        return target / default_name

    if not target.suffix:
        target.mkdir(parents=True, exist_ok=True)
        return target / default_name

    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def MeteoAQ(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    save_path,
    variables=["pm10", "pm2_5", "carbon_monoxide", "carbon_dioxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"],
):
    """
    Retrieve hourly and current air quality data from Open-Meteo API.

    Parameters
    ----------
    lat : float
        Latitude of the location.
    lon : float
        Longitude of the location.
    start_date : str
        Start date in format 'YYYY-MM-DD'.
    end_date : str
        End date in format 'YYYY-MM-DD'.
    save_path : str or PathLike
        Destination CSV file path or output directory.

    Returns
    -------
    tuple
        (current_data_dict, hourly_dataframe)
    """
    # --- Setup client with caching & retry ---
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    cache_session.verify = _resolve_verify_setting()
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # --- Define API endpoint & parameters ---
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": variables,
        "current": "european_aqi",
        "start_date": start_date,
        "end_date": end_date
    }

    # --- Request data ---
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone offset: {response.UtcOffsetSeconds()} s")

    # --- Current data ---
    current = response.Current()
    current_european_aqi = current.Variables(0).Value()
    current_data = {
        "time": current.Time(),
        "european_aqi": current_european_aqi
    }

    print(f"\nCurrent time: {current.Time()}")
    print(f"Current European AQI: {current_european_aqi}")

    # --- Hourly data ---
    hourly = response.Hourly()
    hourly_data = {}
    hourly_data["date"] = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left")
    for i in range(len(variables)):
        # print()
        hourly_data[variables[i]] = hourly.Variables(i).ValuesAsNumpy()
    hourly_df = pd.DataFrame(hourly_data)
    save_path = _resolve_meteoaq_save_path(
        save_path,
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
    )
    hourly_df.to_csv(save_path, index=False)
    # print("\nHourly data sample:")
    # print(hourly_df.head())

    return _ok(
        {
            "csv_path": str(save_path),
            "current": current_data,
            "hourly_records": int(len(hourly_df)),
            "variables": variables,
        }
    )

# Single-level data - download by year
def ERA5(
    save_path,
    product_type,
    variables,
    dataset,
    years=["2015", "2016", "2017", "2018", "2019", "2020"],
    months=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
    days=None,
    hours=["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"],
    pressures=["200", "600", "850", "925"],
    KEY=None,
):
    """
    product_type: ["ensemble_mean"] 
    variables: ["10m_u_component_of_wind","10m_v_component_of_wind","2m_temperature","mean_sea_level_pressure","sea_surface_temperature","total_precipitation","geopotential"],
              ["geopotential","specific_humidity","temperature","u_component_of_wind","v_component_of_wind"],
    dataset: "reanalysis-era5-single-levels", 
             "reanalysis-era5-pressure-levels"
    """ 
    URL = "https://cds.climate.copernicus.eu/api"
    try:
        c = cdsapi.Client(url=URL, key=_resolve_key(KEY), verify=_resolve_verify_setting())
    except Exception as exc:
        return _error(f"Failed to initialize CDS client: {exc}")

    save_dir = Path(save_path).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for year in years:
        for month in months:
            if days is None:
                _, num_days = calendar.monthrange(int(year), int(month))
                day_list = [f"{i:02d}" for i in range(1, num_days+1)]
            else:
                day_list = days

            for day in day_list:
                date = datetime(int(year), int(month), int(day))
                date = date.timetuple().tm_yday
                date = f"{date:03d}"
                # for hh in ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]:
                request = {
                    "product_type": product_type, 
                    "variable": variables, 
                    "year": [year],
                    "month": [month],
                    "day": [day],
                    "time": hours,
                    "data_format": "netcdf",
                    "download_format": "unarchived"
                }
                if dataset == 'reanalysis-era5-pressure-levels':
                    request["pressure_level"] = pressures
                output_file = save_dir / f"{year}{date}.nc"

                out_path = Path(output_file)
                if out_path.exists():
                    print(f"Skipping existing {output_file}")
                    continue

                print(f"Downloading {dataset} data for {year}{date}")
                try:
                    c.retrieve(dataset, request, str(out_path))
                except Exception as exc:
                    return _error(f"CDS request failed for {year}-{month}-{day}: {exc}")
                print(f"Successfully downloaded {output_file}")
                downloaded.append(str(out_path))

    if not downloaded:
        return _ok({"message": "No new files downloaded (all existed).", "dataset": dataset})
    return _ok({"dataset": dataset, "files": downloaded})

def CMIP6(
    save_path,
    variable,
    exp_type,
    model_type,
    temporal_resolution,
    levels=None,
    dataset="projections-cmip6",
    years=["2015", "2016", "2017", "2018", "2019", "2020"],
    months=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
    days=None,
    KEY=None,
    data_format: str = "zip",
):

    URL = "https://cds.climate.copernicus.eu/api"
    try:
        c = cdsapi.Client(url=URL, key=_resolve_key(KEY), verify=_resolve_verify_setting())
    except Exception as exc:
        return _error(f"Failed to initialize CDS client: {exc}")

    save_dir = Path(save_path).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    notes = []

    years = [str(y) for y in years]
    months = [str(m).zfill(2) for m in months]
    variable_list = variable if isinstance(variable, (list, tuple)) else [variable]
    # # Validate variables against the allowed CMIP6 list shown in the CDS UI.
    # invalid_vars = [v for v in variable_list if _normalize_var_name(v) not in _CMIP6_ALLOWED_VARIABLES]
    # if invalid_vars:
    #     return _error(
    #         f"Invalid CMIP6 variable(s): {invalid_vars}. Allowed variables match the CDS UI list."
    #     )
    model_list = model_type if isinstance(model_type, (list, tuple)) else [model_type]

    # Validate base keys (excluding year/month to avoid over-filtering).
    validation_status, validation_note = _validate_with_datastore(
        dataset,
        {
            "temporal_resolution": temporal_resolution,
            "experiment": exp_type,
            "variable": variable_list,
            "model": model_list,
        },
        ["temporal_resolution", "experiment", "variable", "model"],
    )
    if validation_status is False:
        return _error(validation_note)
    if validation_note:
        notes.append(validation_note)

    # Fetch constraints once to verify year/month availability.
    constraints, c_note = _constraints_for_request(
        dataset,
        {
            "temporal_resolution": temporal_resolution,
            "experiment": exp_type,
            "variable": variable_list,
            "model": model_list,
        },
    )
    if c_note:
        notes.append(c_note)
    if constraints:
        allowed_years = {str(y) for y in constraints.get("year", [])}
        missing_years = [y for y in years if y not in allowed_years] if allowed_years else []
        if missing_years:
            return _error(
                f"Requested years not available: {missing_years}. "
                f"Available years sample: {sorted(list(allowed_years))[:20]}"
            )
        allowed_months = {str(m).zfill(2) for m in constraints.get("month", [])}
        missing_months = [m for m in months if m not in allowed_months] if allowed_months else []
        if missing_months:
            return _error(
                f"Requested months not available: {missing_months}. "
                f"Available months: {sorted(list(allowed_months))}"
            )

    if temporal_resolution == "monthly":
        request = {
            "temporal_resolution": temporal_resolution,
            "experiment": exp_type,
            "variable": variable_list,
            "model": model_list,
            "year": years,
            "month": months,
            "format": data_format,
        }
        if levels is not None:
            request["level"] = levels

        output_file = save_dir / f"cmip6_{model_list[0]}_{exp_type}_{years[0]}_{years[-1]}_{months[0]}_{temporal_resolution}.zip"
        if output_file.exists():
            print(f"Found existing {output_file}")
            downloaded.append(str(output_file))
        else:
            print(f"Downloading {dataset} data for years {years} months {months}")
            try:
                c.retrieve(dataset, request, str(output_file))
            except Exception as exc:
                return _error(f"CDS request failed: {exc}")
            print(f"Successfully downloaded {output_file}")
            downloaded.append(str(output_file))
    else:
        for year in years:
            for month in months:
                if days is None:
                    _, num_days = calendar.monthrange(int(year), int(month))
                    day_list = [f"{i:02d}" for i in range(1, num_days+1)]
                else:
                    day_list = days

                for day in day_list:
                    date = datetime(int(year), int(month), int(day))
                    date = date.timetuple().tm_yday
                    date = f"{date:03d}"

                    request = {
                        "temporal_resolution": temporal_resolution,
                        "experiment": exp_type,
                        "variable": variable_list,
                        "model": model_list,
                        "year": [year],
                        "month": [month],
                        "day": [day],
                        "format": data_format,
                    }
                    if levels is not None:
                        request["level"] = levels

                    year_dir = save_dir / year
                    year_dir.mkdir(parents=True, exist_ok=True)
                    output_file = year_dir / f"{year}{date}_daily.zip"
                    if output_file.exists():
                        print(f"Found existing {output_file}")
                        downloaded.append(str(output_file))
                        continue
                    print(f"Downloading {dataset} data for {year}{date}")
                    try:
                        c.retrieve(dataset, request, str(output_file))
                    except Exception as exc:
                        return _error(f"CDS request failed for {year}-{month}-{day}: {exc}")
                    print(f"Successfully downloaded {output_file}")
                    downloaded.append(str(output_file))

    if not downloaded:
        return _ok({"message": "No files downloaded (all existed).", "dataset": dataset, "notes": notes})
    return _ok({"dataset": dataset, "files": downloaded, "notes": notes})

def MSWEP(
    save_path,
    years=["2015", "2016", "2017", "2018", "2019", "2020"],
    months=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
    days=None,
    hours=["00", "03", "06", "09", "12", "15", "18", "21"],
):
    save_dir = Path(save_path).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for year in years:
        for month in months:
            if days is None:
                _, num_days = calendar.monthrange(int(year), int(month))
                day_list = [f"{i:02d}" for i in range(1, num_days+1)]
            else:
                day_list = days
            for day in day_list:
                date = datetime(int(year), int(month), int(day))
                date = date.timetuple().tm_yday
                date = f"{date:03d}"
                for hh in hours:
                    file_name = f"{year}{date}.{hh}.nc"
                    cmd = f"rclone sync -v GoogleDrive:MSWEP_V280/Past/3hourly/{file_name}  {save_dir}/{file_name}"
                    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
                    # Print results
                    print("STDOUT:\n", result.stdout)
                    print("STDERR:\n", result.stderr)
                    downloaded.append(str(save_dir / file_name))

    return _ok({"files": downloaded})

if __name__ == "__main__":
    MeteoAQ(lat = 52.52, lon = 13.41, start_date = "2025-09-01", end_date = "2025-09-14")
    ERA5(save_path = "/l/users/thao.nguyen/TropicalCyclones/TCP-Diffusion/demo", 
        product_type = ["ensemble_mean"] , 
        variables =  ["10m_u_component_of_wind","10m_v_component_of_wind","2m_temperature","mean_sea_level_pressure","sea_surface_temperature","total_precipitation","geopotential"],
        dataset = "reanalysis-era5-single-levels",
        years = ["2015"],
        months = ["01"],
        days = ["01"],
        hours = ["00:00", "03:00"],
        KEY = 'e38962d7-33f4-47a7-9d25-e077512fd7ba')
    
    CMIP6(dataset = "projections-cmip6",
          save_path = "/l/users/thao.nguyen/TropicalCyclones/TCP-Diffusion/demo", 
          variable = "precipitation",
          exp_type = "ssp1_2_6",
          temporal_resolution = "daily",
          model_type = "gfdl_esm4",
          years = ["2017"],
          months = ["01", ],
          days = ["03"]
          )
    CMIP6(dataset = "projections-cmip6",
          save_path = "/l/users/thao.nguyen/TropicalCyclones/TCP-Diffusion/demo", 
          variable = "geopotential_height",
          exp_type = "ssp1_2_6",
          temporal_resolution = "monthly",
          model_type = "mcm_ua_1_0",
          years = ["2043"],
          months = ["03",],
          levels = ["10","20"],
          )
    
    MSWEP(save_path = '/home/salmankhan/zThao/demo',
        years = ["2020"],
        months = ["10"],
        days = ["15"],
        hours = ["03"]
    )
