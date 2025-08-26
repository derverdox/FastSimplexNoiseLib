package de.verdox.util;

import jdk.incubator.vector.FloatVector;
import oshi.SystemInfo;
import oshi.hardware.CentralProcessor;

import java.util.HashMap;
import java.util.Map;

public class HardwareUtil {

    public static int getPhysicalProcessorCount() {
        return new SystemInfo().getHardware().getProcessor().getPhysicalProcessorCount();
    }

    public static void getAmountGPUCores() {
        SystemInfo si = new SystemInfo();
    }

    public static int getVectorLaneLength() {
        return FloatVector.SPECIES_PREFERRED.length();
    }

    public static void printCPU() {
        SystemInfo si = new SystemInfo();
        CentralProcessor cpu = si.getHardware().getProcessor();
        boolean x64 = cpu.getProcessorIdentifier().isCpu64bit();


        CacheSizes cacheSizes = readCaches();
        MemorySizes memorySizes = readMemory();

        System.out.println("CPU: " + cpu.getProcessorIdentifier().getName() + " (x" + (x64 ? "64" : "86") + ")");
        System.out.println("> Cores: " + cpu.getPhysicalProcessorCount() + "C / " + cpu.getLogicalProcessorCount() + "T (Base Clock: " + FormatUtil.formatHz(cpu
                .getProcessorIdentifier().getVendorFreq()) + ", Turbo: " + FormatUtil.formatHz(cpu.getMaxFreq()) + ")");
        System.out.println("> " + cacheSizes.l1.toString());
        System.out.println("> " + cacheSizes.l2.toString());
        System.out.println("> " + cacheSizes.l3.toString());
        System.out.println("> " + memorySizes+"("+FormatUtil.formatBytes2(si.getHardware().getMemory().getTotal())+")");
    }

    public static class CacheSizes {
        public CacheSize l1;
        public CacheSize l2;
        public CacheSize l3;
    }

    public record MemoryPart(long byteSize, long clockRate, String type) {
        @Override
        public String toString() {
            return FormatUtil.formatBytes2(byteSize) + " " + type + "-" + FormatUtil.formatHz(clockRate);
        }
    }

    public record MemorySizes(MemoryPart... memoryParts) {
        @Override
        public String toString() {

            Map<MemoryPart, Integer> counter = new HashMap<>();
            for (MemoryPart memoryPart : memoryParts) {
                counter.compute(memoryPart, (k, v) -> v == null ? 1 : v + 1);
            }

            StringBuilder stringBuilder = new StringBuilder();
            counter.forEach((memoryPart, count) -> {
                stringBuilder.append(count + "x " + memoryPart.toString() + " ");
            });
            return stringBuilder.toString();
        }
    }

    public record CacheSize(int cacheLevel, long sizeBytes, short lineSize, byte associativity, boolean shared) {
        @Override
        public String toString() {
            return "L" + cacheLevel + " Cache: " + FormatUtil.formatBytes10(sizeBytes);
        }
    }

    public static MemorySizes readMemory() {
        SystemInfo si = new SystemInfo();
        return new MemorySizes(si.getHardware().getMemory().getPhysicalMemory().stream()
                                 .map(physicalMemory -> new MemoryPart(physicalMemory.getCapacity(), physicalMemory.getClockSpeed(), physicalMemory.getMemoryType()))
                                 .toArray(MemoryPart[]::new));
    }

    /**
     * Liest L1D/L2/L3 per OSHI.
     */
    public static CacheSizes readCaches() {
        SystemInfo si = new SystemInfo();
        CentralProcessor cpu = si.getHardware().getProcessor();
        CacheSizes cs = new CacheSizes();
        for (CentralProcessor.ProcessorCache c : cpu.getProcessorCaches()) {
            switch (c.getLevel()) {

                case 1 -> {
                    if (c.getLevel() == 1) {
                        cs.l1 = new CacheSize(c.getLevel(), c.getCacheSize(), c.getLineSize(), c.getAssociativity(), false);
                    }
                }
                case 2 -> {
                    if (c.getLevel() == 2) {
                        cs.l2 = new CacheSize(c.getLevel(), c.getCacheSize(), c.getLineSize(), c.getAssociativity(), false);
                    }
                }
                case 3 -> {
                    if (c.getLevel() == 3) {
                        cs.l3 = new CacheSize(c.getLevel(), c.getCacheSize(), c.getLineSize(), c.getAssociativity(), true);
                    }
                }
            }
        }
        return cs;
    }
}
