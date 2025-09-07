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
    implementation("com.github.oshi:oshi-core:6.8.3")

    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
    testImplementation("com.aparapi:aparapi:3.0.0")
    testImplementation("com.github.oshi:oshi-core:6.8.3")
}

tasks.test {
    useJUnitPlatform()
}


tasks.test {
    useJUnitPlatform()
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

tasks.withType<JavaExec>(configuration = {
    jvmArgs(
        "--enable-preview",
        "--add-modules", "jdk.incubator.vector",
        "-Dcom.aparapi.verbose=true",
        "-Dcom.aparapi.enableExecutionModeReporting=false",
        "-Dcom.aparapi.opencl.dumpKernel=true"
    )
})

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
    profilers.set(listOf("gc", "stack"))
    jvmArgsAppend.set(
        listOf(
            "--enable-preview",
            "--add-modules", "jdk.incubator.vector"
        )
    )
}