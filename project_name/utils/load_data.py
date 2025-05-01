import mne
import glob
import os
import logging
import re


# setip logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')


def load_sleep_data(data_dir: str,
                    subject_pattern: str = "*",
                    max_subjects: int | None = None,
                    preload_data: bool = False) -> list[mne.io.Raw]:
    """
    Loads Sleep-EDF(+) data (PSG and Hypnogram EDF files) for specified
    subjects.

    :param data_dir: path to the data.
    :param subject_pattern: glob pattern to match subject IDs
    :param max_subjects: maximum number of subjects to load.
    :param preload_data: Whether to load the raw signal data into memory
                         immediately or load only the header initially.
    :return: list of MNE Raw objects, where each Raw object has its
              annotations set from the corresponding Hypnogram EDF file.
    """
    all_raw_data = []
    search_dirs = [os.path.join(data_dir, 'sleep-cassette'),
                   os.path.join(data_dir, 'sleep-telemetry')]

    psg_files_found = []
    for search_dir in search_dirs:
        if os.path.isdir(search_dir):
            # find all PSG files
            base_psg_pattern = os.path.join(search_dir, "*-PSG.edf")
            all_files_in_dir = glob.glob(base_psg_pattern)

            # filter these files based on the pattern
            pattern_prefix = os.path.basename(subject_pattern).replace("*", "")
            if pattern_prefix:
                # corrected filtering logic
                filtered_files = [
                    f for f in all_files_in_dir
                    if os.path.basename(f).startswith(pattern_prefix)
                ]
                psg_files_found.extend(filtered_files)
            else:
                psg_files_found.extend(all_files_in_dir)

    if not psg_files_found:
        logging.warning("No PSG files found matching pattern " +
                        f"'{subject_pattern}' in {search_dirs}")
        return all_raw_data

    # sort files for consistent processing order
    psg_files_found.sort()

    logging.info(f"Found {len(psg_files_found)} potential PSG files " +
                 f"matching '{subject_pattern}'.")

    subjects_loaded = 0
    for psg_filepath in psg_files_found:
        if max_subjects is not None and subjects_loaded >= max_subjects:
            logging.info("Reached max_subjects limit " +
                         f"({max_subjects}). Stopping.")
            break

        psg_filename = os.path.basename(psg_filepath)
        directory = os.path.dirname(psg_filepath)

        # extract the base part of the filename
        base_psg_part = psg_filename.split('-')[0]

        subject_match = re.match(r"(SC|ST)(\d+)", base_psg_part)
        hypno_search_pattern = None

        if subject_match:
            subject_prefix = subject_match.group(1)
            subject_num_str = subject_match.group(2)

            # create a specific search pattern for the hypnogram file
            hypno_search_pattern = os.path.join(
                directory, f"{subject_prefix}{subject_num_str}*-Hypnogram.edf")
            logging.debug("Using specific hypnogram search pattern: " +
                          f"{hypno_search_pattern}")
        else:
            # fallback if regex doesn't match, use first chars
            stable_prefix = base_psg_part[:5]
            hypno_search_pattern = os.path.join(
                directory, f"{stable_prefix}*-Hypnogram.edf")
            logging.warning("Could not reliably extract subject prefix/" +
                            f"number from {base_psg_part}. Using broader " +
                            f"pattern: {hypno_search_pattern}")

        # search for hypnogram files using the constructed pattern
        matched_hypno_files = glob.glob(hypno_search_pattern)

        hypno_filepath = None
        if len(matched_hypno_files) == 1:
            hypno_filepath = matched_hypno_files[0]
            logging.debug("Found matching hypnogram: " +
                          f"{os.path.basename(hypno_filepath)} for " +
                          f"{psg_filename}")
        elif len(matched_hypno_files) == 0:
            logging.warning("No hypnogram file found matching pattern  " +
                            f"'{hypno_search_pattern}' for {psg_filename}. " +
                            "Skipping.")
            continue
        else:
            logging.warning("Multiple hypnogram files found matching " +
                            f"pattern '{hypno_search_pattern}' "
                            f"for {psg_filename}: {matched_hypno_files}. " +
                            "Skipping due to ambiguity.")
            continue

        try:
            logging.info(f"Loading PSG: {psg_filename} " +
                         f"(preload={preload_data})")

            raw = mne.io.read_raw_edf(
                psg_filepath,
                preload=preload_data,
                exclude=('stim', 'marker', 'Marker', 'Light', 'Events/Markers')
            )

            logging.info("Loading Annotations: " +
                         f"{os.path.basename(hypno_filepath)}")
            # load annotations from the corresponding hypnogram EDF file
            annotations = mne.read_annotations(hypno_filepath)

            # set the annotations to the raw object
            raw.set_annotations(annotations, emit_warning=True)

            # store the subject ID in the info structure for later reference
            raw.info['subject_info'] = {'id': base_psg_part}

            # add the successfully loaded Raw object to the list
            all_raw_data.append(raw)
            subjects_loaded += 1
            logging.info(f"Successfully prepared object for {base_psg_part}.")

        except FileNotFoundError:
            logging.error(f"File not found during loading: {psg_filepath} " +
                          f"or {hypno_filepath}. Check paths.")
        except Exception as e:
            logging.error(f"Failed to load or process {psg_filename} or " +
                          f"its annotations: {e}", exc_info=True)

    logging.info(f"Finished loading. Total subjects: {len(all_raw_data)}")
    return all_raw_data
