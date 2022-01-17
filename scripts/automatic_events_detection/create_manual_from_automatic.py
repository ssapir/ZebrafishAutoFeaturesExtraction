import os.path
import pandas as pd
import sys
import re
import xlsxwriter.utility as xls_util

# Important! in order to use this script, you need to "pip install pandas xlrd" to read excel
# Please no hebrew in excel (trying to use encoding didn't work)
# Important2! If fish already exists, its lines will be overridden by current run (to allow fixes)

LAB_PATH = "\\\\ems.elsc.huji.ac.il\\avitan-lab\\Lab-Shared\\Data\\"
FA_FOLDER = "FeedingAssay2020"
AUTO_SUF = "-automatic-whole-movie-frames"
MANUAL_PATTERN = "FA-{0}.xlsx"
METADATA_FOLDER = "RawMovies"
MANUAL_FOLDER = "ManualAnnotations"
METADATA_NAME = "FeedingAssay-IncreasingDensity.xlsx"

# change me
FISH_NAME = "20210602-f9"  # empty - read from argv, otherwise takes name given here. Example: "FA-20200722-f3.xlsx"
IS_SAVING = True


def get_file_paths(fish_name: str):
    metadata_path_ = os.path.join(LAB_PATH, FA_FOLDER, METADATA_FOLDER, METADATA_NAME)  # todo here of in FA2020?
    one_fish_df_path_ = os.path.join(LAB_PATH, FA_FOLDER, fish_name, fish_name + AUTO_SUF + ".csv")
    if not os.path.exists(one_fish_df_path_):
        one_fish_df_path_ = os.path.join(LAB_PATH, FA_FOLDER, fish_name, fish_name + AUTO_SUF + ".xlsx")
    manual_path_ = os.path.join(LAB_PATH, FA_FOLDER, MANUAL_FOLDER, MANUAL_PATTERN.format(fish_name))
    output_df_path_ = os.path.join(LAB_PATH, FA_FOLDER, fish_name, MANUAL_PATTERN.format(fish_name))
    return one_fish_df_path_, output_df_path_, metadata_path_, manual_path_


def read_file(file_full_path):
    if not os.path.isfile(file_full_path):
        print("Missing file {0}.\nFound files: {1}".format(file_full_path, os.listdir(os.path.dirname(file_full_path))))
        sys.exit(1)
    if file_full_path.endswith(".csv"):
        return pd.read_csv(file_full_path, warn_bad_lines=True, error_bad_lines=False)
    elif file_full_path.endswith(".xlsx") or file_full_path.endswith(".xls"):
        return pd.read_excel(file_full_path)
    print("Error. File format not supported for file ", file_full_path)
    return None  # this is an error


def write_file(df, file_full_path):
    if file_full_path.endswith(".csv"):
        df.to_csv(file_full_path, index=False)
    elif file_full_path.endswith(".xlsx") or file_full_path.endswith(".xls"):
        df.to_excel(file_full_path, index=False)
    else:
        print("Error. File format not supported for file ", file_full_path)
        return  # this is an error
    print("Written in ", file_full_path)


def get_files_and_validate(fish_name):
    def regex_rename(col_name):
        if re.match(r"^endFramecutPoint*", col_name):
            return col_name.replace(r'endFramecutPoint', r'endFrame-cutPoint')
        return col_name

    def fix_column_names(one_fish):
        result = one_fish.rename(columns=regex_rename, inplace=False)
        result.rename(columns={'Status0123': 'Status(0/1/2/3)'}, inplace=True)
        return result

    one_fish_df_path, output_df_path_, metadata_path, manual_path = get_file_paths(fish_name)
    one_fish = fix_column_names(read_file(one_fish_df_path))
    metadata = read_file(metadata_path)
    metadata.drop(metadata.filter(regex="Unnamed"), axis=1, inplace=True)
    curr_fish_metadata = None
    for curr in [FISH_NAME, FISH_NAME.replace("-", "_")]:
        if (curr == metadata.fishName).any():
            curr_fish_metadata = metadata[metadata.fishName == curr]
            break
    if curr_fish_metadata is None:
        print("Error. Fish {0} not found in {1}".format(FISH_NAME, metadata_path))
        sys.exit(1)

    if one_fish.empty:
        print("Error. Fish is empty {1}".format(FISH_NAME, one_fish_df_path))
        sys.exit(1)

    # Make sure columns match (otherwise there will be error)
    if not curr_fish_metadata.columns.isin(one_fish.columns).all():
        print(
            "Error! One fish columns should expand fish metadata columns.\nOne fish columns: {0}.\nMetadata columns: "
            "{1}.\nMissing:".format(one_fish.columns, curr_fish_metadata.columns))
        for name in curr_fish_metadata.columns:
            if not one_fish.columns.isin([name]).any():
                print(name)
        sys.exit(1)

    return curr_fish_metadata, one_fish, output_df_path_, manual_path


if __name__ == "__main__":
    if FISH_NAME == "":
        if len(sys.argv) == 2:
            FISH_NAME = sys.argv[1]
        elif len(sys.argv) == 1:  # only script name at index 0
            print("Wrong usage. Please run: {0} <fish_name>. Example: {0} 20200805-f1".format(sys.argv[0]))
            sys.exit(1)

    curr_fish_metadata_df, one_fish_df, output_df_path, manual_path = get_files_and_validate(FISH_NAME)

    # merge metadata to automatic output
    for col_name in curr_fish_metadata_df.columns.drop(['fishName']):
        one_fish_df[col_name] = curr_fish_metadata_df[col_name].values[0]
    one_fish_df.CuttingComments = ""

    is_expanding_previous_manual = False
    if hasattr(one_fish_df, 'matching_manual'):  # some of the events were already detected
        is_expanding_previous_manual = True

    if pd.api.types.is_numeric_dtype(one_fish_df.is_an_event):
        manual = one_fish_df[one_fish_df.is_an_event == 1].copy()
    else:
        manual = one_fish_df[one_fish_df.is_an_event == '1'].copy()

    if is_expanding_previous_manual:  # only expansion: change name and remove prefound events
        if pd.api.types.is_numeric_dtype(manual.matching_manual):
            manual = manual[manual.matching_manual == -1].copy()
        else:
            manual = manual[manual.matching_manual == '-1'].copy()
        manual.drop(['matching_manual', 'is_manually_detected'], axis=1, inplace=True)

    manual.drop(['AnalysisMovieComments', 'is_an_event'], axis=1, inplace=True)

    column_names = ['startFrame', 'endFrame', 'endFrame-cutPoint', 'outcome', 'complexhunt', 'CuttingComments']
    postfix = "-\d+"
    manual_unnamed_df = manual.filter(regex="{0}{1}".format((postfix + "|").join(column_names), postfix))
    if len(manual_unnamed_df.columns) < len(column_names):
        print("Error. Missing Unnamed columns. Expected {0}, Found {1}".format(len(column_names),
                                                                               manual_unnamed_df.columns))
        sys.exit(1)

    # Stack columns
    from_col_ind = 0
    manual_append_df = pd.DataFrame([], columns=column_names)
    while len(manual_unnamed_df.columns) >= (from_col_ind + len(column_names)):
        manual_unnamed_add = manual_unnamed_df.iloc[:, from_col_ind:min(len(manual_unnamed_df.columns),
                                                                        from_col_ind + len(column_names))].dropna(
            how='all')
        print("manual_unnamed_add shape: ", manual_unnamed_add.shape)
        if not manual_unnamed_add.empty:
            # rename columns: remove last -\d part
            manual_unnamed_add.rename(columns=lambda n: "-".join(n.split("-")[:-1]), inplace=True)
            manual_append_df = manual_append_df.append(manual_unnamed_add, ignore_index=True)
        from_col_ind += len(column_names)  # next column set

    manual = manual.append([manual.iloc[0, :]] * (manual_append_df.shape[0] - manual.shape[0]), ignore_index=True)

    # Copy map (number to name, as defined above)
    for col_name, col_ind in zip(column_names, range(0, len(column_names))):
        manual[col_name] = manual_append_df[col_name].values

    manual.drop(manual.filter(regex="Unnamed"), axis=1, inplace=True)
    manual.drop(manual.filter(manual_unnamed_df.columns), axis=1, inplace=True)
    keys = ['startFrame', 'endFrame', 'endFrame-cutPoint']
    manual[keys] = manual[keys].apply(pd.to_numeric)

    if is_expanding_previous_manual:  # only expansion: change name and remove prefound events
        prev_manual_df = read_file(manual_path)
        if prev_manual_df is not None:
            print("diff:", manual.shape, prev_manual_df.shape)
            manual = pd.concat([manual, prev_manual_df], keys=keys).drop_duplicates(subset=keys, keep='first')
            print("diff:", manual.shape, prev_manual_df.shape)
        else:
            output_df_path = output_df_path.rsplit(".", 1)[0] + "-add." + output_df_path.rsplit(".", 1)[1]

    manual.sort_values(['startFrame', 'endFrame'], inplace=True)
    # write_file(manual, output_df_path)

    writer = pd.ExcelWriter(output_df_path, engine='xlsxwriter')
    manual.to_excel(writer, sheet_name='Sheet1', index=False)
    worksheet = writer.sheets['Sheet1']

    # Apply a conditional format to the cell range.
    format1 = writer.book.add_format({"bg_color": "red"})
    format2 = writer.book.add_format({"bg_color": "orange"})
    startFrameLetter = xls_util.xl_col_to_name(manual.columns.get_loc("startFrame"))  # K
    endFrameLetter = xls_util.xl_col_to_name(manual.columns.get_loc("endFrame"))  # L
    endCutFrameLetter = xls_util.xl_col_to_name(manual.columns.get_loc("endFrame-cutPoint"))  # M
    for criteria, form, s, e in zip(["=(${0}2>${1}2)".format(startFrameLetter, endFrameLetter),  # K > L
                                     "=(${0}2>${1}2)".format(endFrameLetter, endCutFrameLetter),  # L > M
                                     "=(${1}2-${0}2 > 5000)".format(startFrameLetter, endFrameLetter),  # L - K > 5000
                                     "=(${1}2-${0}2 > 5000)".format(startFrameLetter, endCutFrameLetter)  # M - K > 5000
                                     ], [format1, format1, format2, format2],
                                    [startFrameLetter, endFrameLetter, startFrameLetter, startFrameLetter],
                                    [endFrameLetter, endCutFrameLetter, endFrameLetter, endCutFrameLetter]):
        worksheet.conditional_format('{1}2:{2}{0}'.format(manual.shape[0] + 1, s, e),
                                     {"type": "formula",
                                      "criteria": criteria,
                                      "format": form})
    for col_name in ['startFrame', 'endFrame', 'endFrame-cutPoint', "outcome"]:
        outcomeLetter = xls_util.xl_col_to_name(manual.columns.get_loc(col_name))
        worksheet.conditional_format('{1}2:{2}{0}'.format(manual.shape[0] + 1, outcomeLetter, outcomeLetter),
                                     {"type": "formula",
                                      "criteria": "=ISBLANK(${0}2)".format(outcomeLetter),
                                      "format": format1})

    # Close the Pandas Excel writer and output the Excel file.
    print(manual)
    print(output_df_path)
    if IS_SAVING:
        writer.save()
