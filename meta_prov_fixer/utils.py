import os
from zipfile import ZipFile
import json
from collections import defaultdict
from tqdm import tqdm
import tarfile
import time
import zipfile
import lzma
from typing import Generator, List, Literal, Union
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


# def normalise_datetime(datetime_str: str) -> str:
#     """
#     Normalises a datetime string (offset naive or aware, with or without 
#     microseconds) making it a UTC-aware ISO 8601 datetime string with no timestamp
#     (i.e. with no microseconds specified). When converting from offset-naive to 
#     offset-aware (where necessary), Italian timezone is assumed.
    
#     param datetime_str (str): Datetime string, possibly as a timestamp.
#     return: UTC-aware ISO 8601 string without microseconds.
#     """
#     datetime_str = str(datetime_str).replace('Z', '+00:00')
#     dt = datetime.fromisoformat(datetime_str)
    
#     # If naive, assume Europe/Rome
#     if dt.tzinfo is None:
#         dt = dt.replace(tzinfo=ZoneInfo("Europe/Rome"))

#     dt_utc = dt.astimezone(timezone.utc) # convert to UTC
#     dt_utc = dt_utc.replace(microsecond=0) # remove microseconds
    
#     return dt_utc.isoformat()

def normalise_datetime(datetime_str: str) -> str:
    """
    Normalises a datetime string (offset naive or aware, with or without 
    microseconds) making it a UTC-aware ISO 8601 datetime string with no timestamp
    (i.e. with no microseconds specified). When converting from offset-naive to 
    offset-aware (where necessary), Italian timezone is assumed. UTC is made explicit 
    as "+00:00" (not "Z"). If input string contains the explicit xsd datatype (^^xsd:string, 
    ^^xsd:dateTime, ^^http://www.w3.org/2001/XMLSchema#dateTime, or ^^http://www.w3.org/2001/XMLSchema#string),
    the substring representing the datatype is silently removed.
    
    param datetime_str (str): Datetime string, possibly as a timestamp.
    return: UTC-aware ISO 8601 string without microseconds.
    """
    datetime_str = datetime_str.replace("^^xsd:dateTime", "")
    datetime_str = datetime_str.replace("^^http://www.w3.org/2001/XMLSchema#dateTime")
    datetime_str = datetime_str.replace("^^xsd:string", "")
    datetime_str = datetime_str.replace("^^http://www.w3.org/2001/XMLSchema#string")

    dt = datetime.fromisoformat(datetime_str)
    
    # If datetime is naive, assume Europe/Rome
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("Europe/Rome"))
    
    # Convert to UTC and strip microseconds
    dt_utc = dt.astimezone(timezone.utc).replace(microsecond=0)
    
    # # to Format with Z:
    # return dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
    return dt_utc.isoformat()


def get_provenance_graph(entity_iri: str, data_root: str) -> dict:
    """
    Uses the entity's IRI (i.e. its OMID) and finds the exact 
    path of the file storing its provenance graph in a subdirectory of data_root. 
    Then, it reads the file and returns the provenance graph as a dictionary.
    
    param entity_iri: The IRI of the entity whose provenance graph is to be retrieved.
    param data_root: The path to the root directory storing the provenance data.
    return: The provenance graph of the entity as a dictionary.
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
    Iterates over the files in any given directory storing OpenCitations Meta RDF files 
    (metadata or provenance) and yields the JSON-LD data as a list of dictionaries.
    
    :param data_dir: Path to the directory containing the decompressed provenance archive.
    :param to_read: The type of RDF data to read, i.e. either 'provenance' or 'metadata'.
    :yield: A list of dictionaries representing the content of each JSON-LD file.
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
    Extracts only the specified folders and their subfolders from .tar, .zip, or .xz archives.

    :param archive_path: Path to the archive file
    :param output_dir: Directory where files will be extracted
    :param supplier_prefixes: List of supplier prefixes (names of root folders to extract)
    :param entity_type: The type of entity, i.e., one of: 'br', 'id', 'ra', 'ar', 're'
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
    Returns the URI of the resource described by a provenance graph or snapshot by removing the 
    '/prov/se/<counter>' suffix if a snapshot URI is passed, or the '/prov/' suffix if
    a provenance graph IRI (i.e. the graph name) is passed.
    
    :param prov_uri: The provenance URI from which to extract the base URI, i.e. either the URI
        of a snapshot (ending with '/prov/se/<counter>') or the name of a provenance graph (ending with '/prov/').
    :return: The URI of the resource described by the provenance entity, returned as a string.
    """
    if prov_uri.endswith('/prov/'):
        return prov_uri.replace("/prov/", '')
    else:
        return prov_uri.rsplit("/prov/se/", 1)[0]
    
    
def get_seq_num(se_uri: str) -> Union[int, None]:
    """
    Returns as an integer the sequence number of a snapshot (i.e. its counter) if it ends with a number, else returns None.
    param se_uri: the URI of the snapshot entity
    """
    if se_uri[-1].isdigit():
        return int(se_uri.split('/')[-1])
    return None

def remove_seq_num(se_uri:str) -> str:
    """
    Returns the URI of a provenance snapshot without its sequence number (counter).
    E.g. 'https://w3id.org/oc/meta/br/06104375687/prov/se/1' -> 'https://w3id.org/oc/meta/br/06104375687/prov/se/'.
    param se_uri: the URI of the snapshot entity. 
    """
    return se_uri.rsplit('/', 1)[0] + '/'

def get_graph_uri_from_se_uri(se_uri:str) -> str:
    """
    Returns the URI (name) of a provenance named graph starting from the URI of one of its snapshots.
    E.g. 'https://w3id.org/oc/meta/br/06104375687/prov/se/1' -> 'https://w3id.org/oc/meta/br/06104375687/prov/'.
    param se_uri: the URI of the snapshot entity. 
    """
    return se_uri.split('se/', 1)[0]
