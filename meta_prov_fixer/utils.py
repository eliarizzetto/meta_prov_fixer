import os
from zipfile import ZipFile
import json
from collections import defaultdict
from tqdm import tqdm
import tarfile
import time
import zipfile
import lzma
from typing import Generator, List, Literal, Union, Tuple
from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
import warnings
import functools


def log_output_to_file(directory="prov_fix_logs", enabled=False):
    """
    Decorator used to store the output of the methods detecting the errors into files.

    The output is transformed into JSON (with the least possible changes to the original data structures).
    Files are saved inside the 'prov_fix_logs' directory.

    :param directory: Directory where log files are saved.
    :type directory: str
    :param enabled: Whether logging is enabled.
    :type enabled: bool
    :returns: Decorator function.
    """
    def decorator(func):
        def make_json_safe(obj):
            if isinstance(obj, dict):
                return {make_json_safe(k): make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple, set)):
                return [make_json_safe(item) for item in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return make_json_safe(vars(obj))
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            else:
                return str(obj)
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # overwrite 'enabled' with self.log_results if it is specified for the instance
            enabled = getattr(args[0], "log_results", False) if args else enabled 
            result = func(*args, **kwargs)

            if enabled:
                cls_name = args[0].__class__.__name__ if args else "UnknownClass"
                func_name = func.__name__
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                os.makedirs(directory, exist_ok=True)

                # Extract instance attributes safely (if exists)
                instance_attrs = {}
                if args:
                    instance = args[0]
                    try:
                        instance_attrs = make_json_safe(vars(instance))
                    except TypeError:
                        instance_attrs = {}

                input_data = {
                    "timestamp": datetime.now().isoformat(),
                    "class": cls_name,
                    "function": func_name,
                    "input": {
                        "instance_attributes": instance_attrs,
                        "args": make_json_safe(args[1:] if args else []),
                        "kwargs": make_json_safe(kwargs),
                    }
                }

                lines = [json.dumps(input_data, ensure_ascii=False) + '\n']

                if isinstance(result, (list, tuple, set)):
                    lines.extend(
                        json.dumps(make_json_safe(item), ensure_ascii=False) + '\n'
                        for item in result
                    )
                else:
                    lines.append(json.dumps(make_json_safe(result), ensure_ascii=False) + '\n')

                filepath = os.path.join(directory, f"{cls_name}_{func_name}_{timestamp}.jsonl")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.writelines(lines)

            return result
        return wrapper
    return decorator


def normalise_datetime(datetime_str: str) -> str:
    """
    Normalise a datetime string (offset naive or aware, with or without microseconds) making it a UTC-aware ISO 8601 datetime string with no microseconds specified.

    When converting from offset-naive to offset-aware (where necessary), Italian timezone is assumed. UTC is made explicit as 'Z' (not '+00:00'). If input string contains the explicit xsd datatype (^^xsd:string, ^^xsd:dateTime, ^^http://www.w3.org/2001/XMLSchema#dateTime, or ^^http://www.w3.org/2001/XMLSchema#string), the substring representing the datatype is silently removed.

    :param datetime_str: Datetime string, possibly as a timestamp.
    :type datetime_str: str
    :returns: UTC-aware ISO 8601 string without microseconds.
    :rtype: str
    """
    datetime_str = datetime_str.replace("^^xsd:dateTime", "")
    datetime_str = datetime_str.replace("^^http://www.w3.org/2001/XMLSchema#dateTime", "")
    datetime_str = datetime_str.replace("^^xsd:string", "")
    datetime_str = datetime_str.replace("^^http://www.w3.org/2001/XMLSchema#string", "")

    dt = datetime.fromisoformat(datetime_str)
    
    # If datetime is naive, assume Europe/Rome
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("Europe/Rome"))
    
    # Convert to UTC and strip microseconds
    dt_utc = dt.astimezone(timezone.utc).replace(microsecond=0)

    # return dt_utc.isoformat()   # this formats the date with "+00:00" instead of "Z"
    # # to Format with Z:
    return dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')


def get_provenance_graph(entity_iri: str, data_root: str) -> dict:
    """
    Use the entity's IRI (i.e. its OMID) to find the exact path of the file storing its provenance graph in a subdirectory of ``data_root``. Reads the file and returns the provenance graph as a dictionary.

    :param entity_iri: The IRI of the entity whose provenance graph is to be retrieved.
    :type entity_iri: str
    :param data_root: The path to the root directory storing the provenance data.
    :type data_root: str
    :returns: The provenance graph of the entity as a dictionary.
    :rtype: dict
    """
    digits = entity_iri.split('/')[-1] 
    supplier_prefix = digits[:digits.find('0', 1) + 1]
    sequential_number = int(digits.removeprefix(supplier_prefix))
    
    for dirpath, _, _ in os.walk(data_root):
        if os.path.basename(dirpath) == supplier_prefix:
            dir1_path = os.path.join(data_root, dirpath)
            for subdir in sorted(os.listdir(dir1_path), key=lambda x: int(x)):
                if sequential_number < int(subdir):
                    dir2_path = os.path.join(dir1_path, subdir)
                    for subsubdir in sorted([d for d in os.listdir(dir2_path) if d.isdigit()], key=lambda x: int(x)):
                        if sequential_number <= int(subsubdir):
                            dir3_path = os.path.join(dir2_path, subsubdir)
                            prov_dir_path = os.path.join(dir3_path, 'prov')

                            for f in read_rdf_meta_files(prov_dir_path, 'provenance'):
                                data = f
                                if data:
                                    for obj in data:
                                        if obj['@id'] == entity_iri + '/prov/':
                                            return obj
                            break
                    break
    return None


def read_rdf_meta_files(data_dir: str, to_read:Literal["provenance", "metadata"]) -> Generator[List[dict], None, None]:
    """
    Iterate over the files in any given directory storing OpenCitations Meta RDF files (metadata or provenance) and yield the JSON-LD data as a list of dictionaries.

    :param data_dir: Path to the directory containing the decompressed provenance archive.
    :type data_dir: str
    :param to_read: The type of RDF data to read, i.e. either 'provenance' or 'metadata'.
    :type to_read: Literal["provenance", "metadata"]
    :yields: A list of dictionaries representing the content of each JSON-LD file.
    :rtype: Generator[List[dict], None, None]
    """
    fpaths = set()
    to_read = to_read.lower().strip()
    if to_read not in ["provenance", "metadata"]:
        raise ValueError("to_read argument must be either 'provenance' or 'metadata'. ")

    if to_read == 'provenance':
        for dirpath, _, filenames in os.walk(data_dir):
            if os.path.basename(dirpath) == 'prov':
                for fn in filenames:
                    fpaths.add(os.path.join(dirpath,fn))
    elif to_read == 'metadata':
        for dirpath, _, filenames in os.walk(data_dir):
            if os.path.basename(dirpath).isnumeric() and os.path.basename(dirpath) != 'prov':
                for fn in filenames:
                    fpaths.add(os.path.join(dirpath,fn))

    for fp in fpaths:
        if fp.endswith('.zip'):
            with ZipFile(fp) as archive:  # Handle zip files
                for f in archive.filelist:
                    if f.filename.endswith('.json'):
                        with archive.open(f.filename) as f:
                            data = json.load(f)
                            yield data
        elif fp.endswith('.json.xz'):  # Handle lzma2 (.xz) files
            with lzma.open(fp, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                    yield data


def extract_specific_folders(archive_path, output_dir, supplier_prefixes: list, entity_type: str):
    """
    Extract only the specified folders and their subfolders from .tar, .zip, or .xz archives.

    :param archive_path: Path to the archive file.
    :type archive_path: str
    :param output_dir: Directory where files will be extracted.
    :type output_dir: str
    :param supplier_prefixes: List of supplier prefixes (names of root folders to extract).
    :type supplier_prefixes: list
    :param entity_type: The type of entity, i.e., one of: 'br', 'id', 'ra', 'ar', 're'.
    :type entity_type: str
    :returns: None
    """
    start_time = time.time()  # Start timing
    print(f"Opening archive: {archive_path}")

    folders_to_extract = [os.path.join(entity_type, sp) for sp in supplier_prefixes]

    if archive_path.endswith(('.tar', '.tar.gz', '.tar.bz2', '.tar.xz', '.xz')):
        with tarfile.open(archive_path, "r:*") as tar:
            members = tar.getmembers()
            members_to_extract = [m for m in members if any(os.path.normpath(m.name).startswith(folder) for folder in folders_to_extract)]
            
            print(f"Total files/folders to extract: {len(members_to_extract)}")
            if not members_to_extract:
                print("No matching folders found in the tarball.")
                return
            
            for member in tqdm(members_to_extract, desc="Extracting", unit="file"):
                tar.extract(member, path=output_dir)
    
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            members_to_extract = [m for m in members if any(os.path.normpath(m).startswith(folder) for folder in folders_to_extract)]
            
            print(f"Total files/folders to extract: {len(members_to_extract)}")
            if not members_to_extract:
                print("No matching folders found in the zip archive.")
                return
            
            for member in tqdm(members_to_extract, desc="Extracting", unit="file"):
                zip_ref.extract(member, output_dir)
    
    else:
        print("Unsupported file format.")
        return
    
    end_time = time.time()
    print(f"\nExtraction completed in {end_time - start_time:.2f} seconds.")
    print(f"Extracted files are saved in: {output_dir}")


def get_described_res_omid(prov_uri: str) -> str:
    """
    Return the URI of the resource described by a provenance graph or snapshot by removing the 
    '/prov/se/<counter>' suffix if a snapshot URI is passed, or the '/prov/' suffix if a provenance graph IRI (i.e. the graph name) is passed.

    :param prov_uri: The provenance URI from which to extract the base URI, i.e. either the URI of a snapshot (ending with '/prov/se/<counter>') or the name of a provenance graph (ending with '/prov/').
    :type prov_uri: str
    :returns: The URI of the resource described by the provenance entity.
    :rtype: str
    """
    if prov_uri.endswith('/prov/'):
        return prov_uri.replace("/prov/", '')
    else:
        return prov_uri.rsplit("/prov/se/", 1)[0]
    
    
def get_seq_num(se_uri: str) -> Union[int, None]:
    """
    Return as an integer the sequence number of a snapshot (i.e. its counter) if it ends with a number, else returns None.

    :param se_uri: The URI of the snapshot entity.
    :type se_uri: str
    :returns: Sequence number as integer, or None.
    :rtype: Union[int, None]
    """
    if se_uri[-1].isdigit():
        return int(se_uri.split('/')[-1])
    return None

def remove_seq_num(se_uri:str) -> str:
    """
    Return the URI of a provenance snapshot without its sequence number (counter).

    Example:

    .. code-block:: python

        'https://w3id.org/oc/meta/br/06104375687/prov/se/1' -> 'https://w3id.org/oc/meta/br/06104375687/prov/se/'

    :param se_uri: The URI of the snapshot entity.
    :type se_uri: str
    :returns: URI without sequence number.
    :rtype: str
    """
    return se_uri.rsplit('/', 1)[0] + '/'

def get_graph_uri_from_se_uri(se_uri:str) -> str:
    """
    Return the URI (name) of a provenance named graph starting from the URI of one of its snapshots.

    Example:

    .. code-block:: python

        'https://w3id.org/oc/meta/br/06104375687/prov/se/1' -> 'https://w3id.org/oc/meta/br/06104375687/prov/'

    :param se_uri: The URI of the snapshot entity.
    :type se_uri: str
    :returns: The URI of the provenance named graph.
    :rtype: str
    """
    return se_uri.split('se/', 1)[0]

def get_previous_meta_dump_uri(meta_dumps_pub_dates, dt:str)-> str:
    """
    Return the DOI of the OpenCitations Meta dump that was published before the given date.

    Example of a well-formed ``meta_dumps_pub_dates`` register:

    .. code-block:: python

        meta_dumps_pub_dates = [
            ('2022-12-19', 'https://doi.org/10.6084/m9.figshare.21747536.v1'),
            ('2022-12-20', 'https://doi.org/10.6084/m9.figshare.21747536.v2'),
            ('2023-02-15', 'https://doi.org/10.6084/m9.figshare.21747536.v3'),
            ('2023-06-28', 'https://doi.org/10.6084/m9.figshare.21747536.v4'),
            ('2023-10-26', 'https://doi.org/10.6084/m9.figshare.21747536.v5'),
            ('2024-04-06', 'https://doi.org/10.6084/m9.figshare.21747536.v6'),
            ('2024-06-17', 'https://doi.org/10.6084/m9.figshare.21747536.v7'),
            ('2025-02-02', 'https://doi.org/10.6084/m9.figshare.21747536.v8'),
            ('2025-07-10', 'https://doi.org/10.5281/zenodo.15855112')
        ]

    :param meta_dumps_pub_dates: List of tuples (date, DOI) for Meta dumps.
    :type meta_dumps_pub_dates: list
    :param dt: A date string in ISO format (YYYY-MM-DD).
    :type dt: str
    :returns: The DOI of the previous Meta dump.
    :rtype: str
    """

    # meta_dumps_pub_dates = sorted([(date.fromisoformat(d), doi) for d, doi in meta_dumps_pub_dates], key=lambda x: x[0])
    d = date.fromisoformat(dt.strip()[:10])
    res = None
    for idx, t in enumerate(meta_dumps_pub_dates):
        if d <= t[0]:
            pos = idx-1 if ((idx-1) >= 0) else 0 # if dt predates the publication date of the absolute first Meta dump, assign the first Meta dump
            prim_source = meta_dumps_pub_dates[pos][1]
            res = prim_source
            break
    if res:
        return res
    else:
        warnings.warn(f'[get_previous_meta_dump_uri]: {dt} follows the publication date of the latest Meta dump. The register of published dumps might need to be updated!') 
        return meta_dumps_pub_dates[-1][1] # picks latest dump in the register


def validate_meta_dumps_pub_dates(meta_dumps_register:List[Tuple[str, str]]):
    """
    Validate the register of published OpenCitations Meta dumps.

    Example of a well-formed register:

    .. code-block:: python

        [
            ('2022-12-19', 'https://doi.org/10.6084/m9.figshare.21747536.v1'),
            ('2022-12-20', 'https://doi.org/10.6084/m9.figshare.21747536.v2'),
            ('2023-02-15', 'https://doi.org/10.6084/m9.figshare.21747536.v3'),
            ('2023-06-28', 'https://doi.org/10.6084/m9.figshare.21747536.v4'),
            ('2023-10-26', 'https://doi.org/10.6084/m9.figshare.21747536.v5'),
            ('2024-04-06', 'https://doi.org/10.6084/m9.figshare.21747536.v6'),
            ('2024-06-17', 'https://doi.org/10.6084/m9.figshare.21747536.v7'),
            ('2025-02-02', 'https://doi.org/10.6084/m9.figshare.21747536.v8'),
            ('2025-07-10', 'https://doi.org/10.5281/zenodo.15855112')
        ]

    :param meta_dumps_register: List of tuples (date, DOI) for Meta dumps.
    :type meta_dumps_register: List[Tuple[str, str]]
    :returns: None
    """
    meta_dumps_register = sorted(meta_dumps_register, key=lambda x: datetime.strptime(x[0], r'%Y-%m-%d'))
    if len(meta_dumps_register) < 9: # number of published Meta dumps at the time of writing this code (2025-07-30)
        raise ValueError("[validate_meta_dumps_pub_dates]: The list of published Meta dumps is incomplete and must be updated.")
    
    # warn if the last date is more than 2 months ago
    last_date = datetime.strptime(meta_dumps_register[-1][0], r'%Y-%m-%d')
    if (datetime.now() - last_date).days > 60:
        warnings.warn(f"[validate_meta_dumps_pub_dates]: The latest Meta dump in the register ({last_date.strftime(r'%Y-%m-%d')}) is more than 2 months old. Make sure to update the register with the latest publication dates and DOIs!")
    
    for index, item in enumerate(meta_dumps_register):
        # Check type and length
        if not isinstance(item, tuple):
            raise ValueError(f"[validate_meta_dumps_pub_dates]: Item at index {index} is not a tuple: {item}")
        if len(item) != 2:
            raise ValueError(f"[validate_meta_dumps_pub_dates]: Tuple at index {index} does not have 2 elements: {item}")

        date_str, url = item

        # Validate ISO date
        try:
            datetime.strptime(date_str, r'%Y-%m-%d')
        except ValueError:
            raise ValueError(f"[validate_meta_dumps_pub_dates]: Invalid ISO date at index {index}: {date_str}")

        # Validate URL
        parsed_url = urlparse(url)
        if not (parsed_url.scheme in ('http', 'https') and parsed_url.netloc):
            raise ValueError(f"[validate_meta_dumps_pub_dates]: Invalid URL at index {index}: {url}")
