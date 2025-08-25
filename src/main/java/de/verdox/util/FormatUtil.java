package de.verdox.util;

import java.util.Locale;

public class FormatUtil {
    public static String formatBytes2(long bytes) {
        if (bytes < 1024) {
            return bytes + " B";
        }
        String[] units = {"KiB", "MiB", "GiB", "TiB", "PiB", "EiB"};
        int exp = (int) (Math.log(bytes) / Math.log(1024));
        double value = bytes / Math.pow(1024, exp);
        return String.format(Locale.US, "%.2f %s", value, units[exp - 1]);
    }

    public static String formatBytes10(long bytes) {
        if (bytes < 1000) {
            return bytes + " B";
        }
        String[] units = {"KB", "MB", "GB", "TB", "PB", "EB"};
        int exp = (int) (Math.log(bytes) / Math.log(1000));
        double value = bytes / Math.pow(1000, exp);
        return String.format(Locale.US, "%.2f %s", value, units[exp - 1]);
    }

    public static String formatHz(long frequency) {
        if (frequency < 1000) {
            return frequency + " B";
        }
        String[] units = {"kHz", "MHz", "Ghz", "Thz", "Phz", "Ehz"};
        int exp = (int) (Math.log(frequency) / Math.log(1000));
        double value = frequency / Math.pow(1000, exp);
        return String.format(Locale.US, "%.2f %s", value, units[exp - 1]);
    }
}
