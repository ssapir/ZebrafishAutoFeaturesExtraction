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

# change me
# FISH_NAME = "20210502-f7"  # empty - read from argv, otherwise takes name given here. Example: "FA-20200722-f3.xlsx"
FISH_NAME = "20210822-f3"  # empty - read from argv, otherwise takes name given here. Example: "FA-20200722-f3.xlsx"
IS_SAVING = True
is_automatic = True


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


def fix_column_names(one_fish):
    def regex_rename(col_name):
        if re.match(r"^endFramecutPoint*", col_name):
            return col_name.replace(r'endFramecutPoint', r'endFrame-cutPoint')
        return col_name
    result = one_fish.rename(columns=regex_rename, inplace=False)
    result.rename(columns={'Status0123': 'Status(0/1/2/3)'}, inplace=True)
    return result


if __name__ == "__main__":
    if FISH_NAME == "":
        if len(sys.argv) == 2:
            FISH_NAME = sys.argv[1]
        elif len(sys.argv) == 1:  # only script name at index 0
            print("Wrong usage. Please run: {0} <fish_name>. Example: {0} 20200805-f1".format(sys.argv[0]))
            sys.exit(1)

    if is_automatic:
        input_df_path = os.path.join(LAB_PATH, FA_FOLDER, FISH_NAME, FISH_NAME + AUTO_SUF + ".csv")  # todo change to xlsx?
        output_df_path = os.path.join(LAB_PATH, FA_FOLDER, FISH_NAME, FISH_NAME + AUTO_SUF + ".xlsx")
    else:
        input_df_path = os.path.join(LAB_PATH, FA_FOLDER, FISH_NAME, MANUAL_PATTERN.format(FISH_NAME))
        output_df_path = os.path.join(LAB_PATH, FA_FOLDER, FISH_NAME, MANUAL_PATTERN.format(FISH_NAME + "-new"))

    curr_df_path = fix_column_names(read_file(input_df_path))

    if curr_df_path.empty:
        print("Error. can't read file {0}".format(output_df_path))
        sys.exit(1)

    # write_file(manual, output_df_path)

    writer = pd.ExcelWriter(output_df_path, engine='xlsxwriter')
    curr_df_path.to_excel(writer, sheet_name='Sheet1', index=False)
    worksheet = writer.sheets['Sheet1']
    # Apply a conditional format to the cell range.
    format1 = writer.book.add_format({"bg_color": "red"})
    format2 = writer.book.add_format({"bg_color": "orange"})

    if is_automatic:
        names = ["-" + str(i) for i in range(1, 4)]
    else:
        names = [""]
    for i in names:
        startFrameLetter = xls_util.xl_col_to_name(curr_df_path.columns.get_loc("startFrame" + i))  # K
        endFrameLetter = xls_util.xl_col_to_name(curr_df_path.columns.get_loc("endFrame" + i))  # L
        endCutFrameLetter = xls_util.xl_col_to_name(curr_df_path.columns.get_loc("endFrame-cutPoint" + i))  # M
        for criteria, form, s, e in zip(["=(${0}2>${1}2)".format(startFrameLetter, endFrameLetter),  # K > L
                                         "=(${0}2>${1}2)".format(endFrameLetter, endCutFrameLetter),  # L > M
                                         "=(${1}2-${0}2 > 5000)".format(startFrameLetter, endFrameLetter),  # L - K > 5000
                                         "=(${1}2-${0}2 > 5000)".format(startFrameLetter, endCutFrameLetter)  # M - K > 5000
                                         ], [format1, format1, format2, format2],
                                        [startFrameLetter, endFrameLetter, startFrameLetter, startFrameLetter],
                                        [endFrameLetter, endCutFrameLetter, endFrameLetter, endCutFrameLetter]):
            worksheet.conditional_format('{1}2:{2}{0}'.format(curr_df_path.shape[0] + 1, s, e),
                                         {"type": "formula",
                                          "criteria": criteria,
                                          "format": form})
    for col_name in ['startFrame', 'endFrame', 'endFrame-cutPoint', "outcome"]:
        outcomeLetter = xls_util.xl_col_to_name(curr_df_path.columns.get_loc(col_name))
        worksheet.conditional_format('{1}2:{2}{0}'.format(curr_df_path.shape[0] + 1, outcomeLetter, outcomeLetter),
                                     {"type": "formula",
                                      "criteria": "=ISBLANK(${0}2)".format(outcomeLetter),
                                      "format": format1})
    # Close the Pandas Excel writer and output the Excel file.
    print(curr_df_path)
    print(output_df_path)
    if IS_SAVING:
        writer.save()
