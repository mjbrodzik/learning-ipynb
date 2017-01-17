#!/usr/bin/env python
"""Library for handling hypsometry-format data for CHARIS melt modelling.

Most CHARIS melt modelling is performed on
hypsometrically-formatted data.  This module provides a mechanism
for reading, writing, displaying and storing this data.

We define a CHARIS "hypsometry" object as having 2 parts, a list
of comments that can be used to store data provenance and history
of the actual data, and a python pandas "DataFrame" to store and
manipulate the actual data.

"""
from __future__ import print_function
import csv   # noqa
import datetime as dt   # noqa
import glob   # noqa
import matplotlib.dates as mdates  # noqa
import matplotlib.pyplot as plt  # noqa
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa
import os  # noqa
import numpy as np   # noqa
import pandas as pd   # noqa
import re   # noqa
import sys   # noqa


class Hypsometry():
    """Hypsometry class manages CHARIS hypsometry data and metadata.

    Public attributes:

    - comments: List of strings with metadata about the data, can
      include provenance, units, history or any other relevant
      description of the data contents.

    - data: Pandas DataFrame with hypsometry data, with columns
      at regular elevation bands and rows representing dates.

    """
    comments = []
    data = pd.DataFrame()

    def __init__(self, filename=None, comments=None, data=None, verbose=False):
        """Initialize a Hypsometry data object.

        Initializes a Hypsometry data object with input comments and data, or
        reads the contents of the input filename.

        Hypsometry data read from various other (IDL and python)
        systems is currently read as-is, with various versions of
        strings/floats for column names.  This creates problems
        when combining Hypsometry DataFrames with column names
        that should be considered the same, but that pandas
        treats as distinct.  We should consider addressing these
        inconsistencies as best we can in this initializer
        routine.

        Args:
          filename : (optional) filename to open and read from

          comments : list of string metadata describing the data;
            (ignored if filename is input)

          data : pandas DataFrame, with elevations in columns and
            dates in rows; DataFrame index can be ['NoDate'] for
            undated data, or pandas.tseries.index.DatetimeIndex
            if contents are dated;
            (ignored if filename is input)

          verbose: boolean to turn on verbose output to stderr.

        Returns:
          Initialized CHARIS Hypsometry object.

        """
        # Set default state information
        self.comments = []
        self.data = pd.DataFrame()

        if filename is not None:
            self.read(filename, verbose=verbose)
        else:
            # Accept any input state values
            if comments is not None:
                self.comments = self.comments + comments

            if data is not None:
                self.data = data.copy()

        # Strip eol characters from the comments
        if self.comments is not None:
            self.comments = [line.rstrip('\r\n') for line in self.comments]

        if verbose:
            print("> %s : initialized new hypsometry object" % (__name__),
                  file=sys.stderr)

    def read(self, filename, verbose=False):
        """Reads ASCII hypsometry file into the current object.

        Comments read from the filename will be appended to object's comments.
        Data read from the filename will overwrite object's data.

        Raises any file IO errors encountered during the read procedure.

        Assumes the filename is ASCII and follows the convention:

        1) 0 or more comment lines, beginning with '#'
        2) 1 line with the number of elevation bands in this file, e.g. NN
        3) 1 line with whitespace-delimited list of lower bounds of each
           elevation band
        4) Data records by date, of the form:
           yyyy mm dd ddd [whitespace delimited list of data of NN data values]

        Args:
          filename: string name of hypsometry filename to read

          verbose: boolean to turn on verbose output to stderr.

        """
        # Import the whole file once and
        # Parse the file comments and elevation data first.
        # Elevation data will be used to set pandas column labels
        lines = open(filename).readlines()
        num_comments = 0
        regex_leading_comment = re.compile(r'#')
        for line in lines:
            part = regex_leading_comment.match(line)
            if part is None:
                break
            else:
                num_comments += 1
                self.comments.append(line.rstrip('\r\n'))

        # Check for no data, indicated by a blank line at the end of comments
        if lines[num_comments + 1] != '\n':

            # Now use the elevation information to set up column headers
            col_names = ['yyyy',
                         'mm', 'dd', 'doy'] + lines[num_comments + 1].split()

            # Now read the data part of the file into a DataFrame Use
            # header=None in order to no use anything in the file for
            # the header, and to pass in col_names list for column
            # headers instead.  Tell it to skip the comments and the
            # two leading rows before reading real data
            self.data = pd.read_table(filename, sep="\s+", skiprows=num_comments+2,
                                      header=None, names=col_names,
                                      index_col='doy')

            # For input data with dates Use the yyyy, mm, dd columns
            # to make a time series index for this DataFrame:
            # otherwise the data are not date-specific, so don't try
            # to make datetime index
            if 999 != self.data.index[0]:
                date_list = []
                for (yyyy, mm, dd) in zip(self.data['yyyy'].values,
                                          self.data['mm'].values,
                                          self.data['dd'].values):
                    date_list.append(dt.date(yyyy, mm, dd))
                    dates = pd.to_datetime(date_list)
                self.data['Date'] = dates
            else:
                self.data['Date'] = ['NoDate']

            self.data.set_index(['Date'], inplace=True)
            self.data.drop(['yyyy', 'mm', 'dd'], axis=1, inplace=True)

        if verbose:
            print ("> %s : read hypsometry data from %s" % (__name__, filename),
                   file=sys.stderr)
            print ("> %s : %d comments." % (__name__, len(self.comments)),
                   file=sys.stderr)
            print ("> %s : %d dates." % (__name__, len(self.data.index)),
                   file=sys.stderr)
            print ("> %s : %d elevations." % (__name__, len(self.data.columns)),
                   file=sys.stderr)

    def write(self, filename, decimal_places=6, verbose=False):
        """Writes object contents to ASCII hypsometry file.

        See read method for CHARIS hypsometry file format convention.

        If object comments do not begin with '#', then they will
        be prepended by this character in the output stream.

        If filename directory parent (one level only) doesn't exist,
        it will be created.

        Args:
          filename: string name of hypsometry filename to write

          decimal_places: number of decimal places for formatted data values
            in output file

          verbose: boolean to turn on verbose output to stderr.

        Returns:
          True if successful

        """
        try:
            fh = open(filename, 'w')
        except IOError as e:
            if e.errno == 2:
                # Create output path if it's not already there.
                path = os.path.dirname(filename)
                if not os.path.isdir(path):
                    os.mkdir(path)
                    if verbose:
                        print("> %s : creating new directory=%s" % (__name__, path),
                              file=sys.stderr)
                    fh = open(filename, 'w')
            else:
                print("> %s : raising other IOError=%s" % (__name__, e),
                      file=sys.stderr)
                raise
        except:
            raise

        # Make a copy of the data frame to work on here.
        # We will sort it by elevation, and add any date
        # columns to the beginning before writing to the output file
        tmp = self.data.copy(deep=True)

        # Coerce the columns to floats so they sort numerically
        # Sort the data frame columns by elevation value
        # so they are written to the output file by increasing elevation
        tmp.columns = tmp.columns.astype('float')
        tmp.sort_index(axis=1, inplace=True)

        # Write any comments first, one comment per line
        regex_leading_comment = re.compile(r'#')
        regex_trailing_newline = re.compile(r'\n$')
        for line in self.comments:
            prefix = ''
            suffix = ''
            if regex_leading_comment.match(line) is None:
                prefix = '# '
            if regex_trailing_newline.match(line) is None:
                suffix = '\n'
            fh.write(prefix + line + suffix)

        # Write the number of columns (elevations) in the DataFrame
        fh.write(str(len(tmp.columns)) + '\n')

        # Write the elevations for each column, whitespace-separated
        fh.write(' '.join([str(col) for col in tmp.columns]) + '\n')

        # Close the file so DataFrame.to_csv method can now append to it
        fh.close()

        # Only write the data part if there are data to write
        if len(tmp.columns) > 0:

            # Make a temporary copy of the DataFrame,
            # and if the data have dates (some do not),
            # add back columns for year, month, day and doy
            if tmp.index[0] != 'NoDate':
                tmp['yyyy'] = tmp.index.year
                tmp['mm'] = tmp.index.month
                tmp['dd'] = tmp.index.day
                tmp['doy'] = tmp.index.dayofyear
            else:
                tmp['yyyy'] = [9999]
                tmp['mm'] = [99]
                tmp['dd'] = [99]
                tmp['doy'] = [999]
            tmp = tmp.reindex_axis(['yyyy', 'mm', 'dd', 'doy'] +
                                   list(tmp.columns[:-4]), axis=1)

            # Sort the columns by elevation

            # Aggravating behavior: if I put a numeral in front of
            # the format decimal place, it will put double-quotes
            # around every floating-point value in the output file no
            # matter what combination of the to_csv switches I use.
            format = "%." + str(decimal_places) + "f"
            tmp.to_csv(filename, mode='a', header=False, index=False, sep=" ",
                       float_format=format, quoting=csv.QUOTE_NONE)

            # I'm not convinced that this actually frees the tmp variable
            del tmp

        if verbose:
            print("> %s : wrote hypsometry data to %s" % (__name__, filename),
                  file=sys.stderr)
            print("> %s : %d comments." % (__name__, len(self.comments)),
                  file=sys.stderr)
            print("> %s : %d dates." % (__name__, len(self.data.index)),
                  file=sys.stderr)
            print("> %s : %d elevations." % (__name__, len(self.data.columns)),
                  file=sys.stderr)

        return True

    def print(self, file=sys.stderr):
        """Prints hypsometry comments and first 5 rows of dataFrame.

        This print function relies on "from __future__ import print_function"
        If the caller has not done this import, nothing will be printed.
        I would like to figure out how to make it behave properly no matter
        what.

        Args:
          file: output file stream

        """

        for line in self.comments:
            print(line, file=file)

        print(self.data.head(), file=file)

    def imshow(self, ax, title=None, cmap="Greys_r",
               xlabel="Date",
               dateFormat="%b",
               ylabel="Elevation ($m$)",
               vmin=None,
               vmax=None,
               verbose=False):
        """Displays the hypsometry data as a color image in the axes.

        Args:
          ax: Axes object to add the data image to

          title: string, image title

          cmap: Colormap

          xlabel, ylabel: strings, axes labels

          dateFormat: string, date format string for X axis labels

          vmin, vmax: min/max value to display
            Default values are min/max of data

          verbose: boolean to turn on verbose output to stderr.

        Returns:
          AxesImage

        """
        if verbose:
            print("> %s : begin." % (__name__), file=sys.stderr)

        if title:
            ax.set_title(title)

        if not vmin:
            vmin = np.amin(self.data.values)

        if not vmax:
            vmax = np.amax(self.data.values)

        # Create xlims as sequence of floats for each date
        x_lims = mdates.date2num([self.data.index[0].to_pydatetime(),
                                  self.data.index[-1].to_pydatetime()])
        y_lims = [float(self.data.columns[0]),
                  float(self.data.columns[-1])]
        extent = [x_lims[0], x_lims[1], y_lims[0], y_lims[1]]

        im = ax.imshow(np.rot90(self.data), aspect="auto", extent=extent,
                       vmin=vmin,
                       vmax=vmax,
                       cmap=cmap)

        # Create divider for existing axes instance
        divider = make_axes_locatable(ax)

        # Append axes to the right of ax, with 7% width of ax
        cax = divider.append_axes("right", size="7%", pad=0.1)

        # Create colorbar in the appended axes
        plt.colorbar(im, cax=cax)

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter(dateFormat))

        return ax

    def barplot(self, ax, index, title=None, color='b',
                xlabel="Value",
                ylabel="Elevation ($m$)",
                verbose=False):
        """Displays the hypsometry data as a horizontal barplot in the axes.

        Args:
          ax: Axes object to add the data image to

          title: string, image title

          color: bar color

          xlabel, ylabel: strings, axes labels

          verbose: boolean to turn on verbose output to stderr.

        Returns:
          AxesImage

        """
        if verbose:
            print("> %s : begin." % (__name__), file=sys.stderr)

        if title:
            ax.set_title(title)

        self.data.ix[index].plot(ax=ax, kind='barh', color=color)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_yticks(ax.get_yticks()[::10])
        ax.set_yticklabels(self.data.columns[::10])

        return ax

    def compare(self, other, ignore_comments=False,
                collapse_zero_columns=False, verbose=False):
        """Compares data contents to that of another hypsometry object, for
        equality.

        Args:
          other: another hypsometry object to compare to

          ignore_comments: boolean, if True, only compare data,
            and ignore comments

          collapse_zero_columns: boolean, if True, collapse any columns in
            either DataFrame with all zeroes.

          verbose: boolean to turn on verbose output to stderr.

        Returns:
          True if object data frames are identical, according to
            requested switches.

        """
        # Compare data frames
        if verbose:
            print("> %s : this object" % (__name__), file=sys.stderr)
            self.print()
            print("> %s : compare object" % (__name__), file=sys.stderr)
            other.print()

        if not collapse_zero_columns:
            data_equal = self.data.equals(other.data)
        else:
            # If collapsing zero columns, find and drop any columns in
            # either hypsometry that are all zeroes
            data_equal = self.data.drop(
                self.data.columns[(self.data == 0).all()], axis=1).equals(
                    other.data.drop(
                        other.data.columns[(other.data == 0).all()], axis=1))

        if ignore_comments or not data_equal:
            return data_equal

        # Compare comments
        return self.comments == other.comments

    def data_by_doy(self, verbose=False):
        """Sum hypsometry data by day-of-year.

        Args:
          verbose: boolean to turn on verbose output to stderr.

        Returns:
          Pandas Series object, with the hypsometry DataFrame data summed
          by row (day-of-year)

        """
        if verbose:
            print("> %s : begin." % (__name__), file=sys.stderr)

        return(self.data[self.data.columns].sum(axis=1))
