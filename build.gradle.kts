plugins {
    id("java")
    id("me.champeau.jmh") version "0.7.2"
}

group = "de.verdox"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("com.aparapi:aparapi:3.0.0")

    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.test {
    useJUnitPlatform()
}


tasks.test {
    useJUnitPlatform()
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

tasks.compileJava {
    options.compilerArgs.add("--enable-preview")
    options.compilerArgs.add("--add-modules")
    options.compilerArgs.add("jdk.incubator.vector")
}

tasks.compileTestJava {
    options.compilerArgs.add("--enable-preview")
}

jmh {
    warmupIterations.set(5)
    iterations.set(10)
    fork.set(2)

    // Falls du die Vector API (Incubator/Preview) nutzt:
    // Achtung: Einige JDKs erwarten nur --enable-preview, andere zusätzlich --add-modules
    jvmArgsAppend.set(
        listOf(
            "--enable-preview",
            "--add-modules", "jdk.incubator.vector"
        )
    )
}