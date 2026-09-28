"""Generate JSON-lines workers using the native Doocs types and entrypoints."""
from __future__ import annotations


python_src = 'import contextlib\nimport json\nimport sys\nfrom typing import *\nfrom collections import *\nfrom functools import *\nfrom itertools import *\nfrom heapq import *\nfrom bisect import *\nfrom math import *\nimport collections, functools, itertools, heapq, bisect, math, random, string\nimport builtins\n# math\'s two-argument pow must not shadow the builtin (pow(a, b, mod)).\npow = builtins.pow\n# Keep `if __name__ == "__main__":` blocks in the solution from running.\n__name__ = "solution"\nwith contextlib.redirect_stdout(sys.stderr):\n    exec(open("solution.py").read(), globals())\n    candidate = {target}\nfor line in sys.stdin:\n    args = json.loads(line)\n    with contextlib.redirect_stdout(sys.stderr):\n        result = candidate(*args)\n    print(json.dumps(result, allow_nan=False), flush=True)\n'


def render_worker(problem, language):
    """Generate the native JSON-lines runner source (worker.py / runner.<lang>)
    for a (problem, language) flat row."""
    interface = problem["interface"]
    params = interface["parameters"]
    names = ", ".join(f"arg{i}" for i in range(len(params)))
    call = interface["callable"]
    container = interface.get("container")

    if language == "python":
        target = f"{container}().{call}" if container else call
        return python_src.format(target=target)

    if language == "cpp":
        declarations = "\n".join(
            f"auto arg{i} = args.at({i}).get<{p['type'].replace('&', '').replace('const ', '').strip()}>();"
            for i, p in enumerate(params)
        )
        target = f"solution.{call}" if container else call
        instance = f"{container} solution;" if container else ""
        output = "string(1, result)" if interface["return_type"] == "char" else "result"
        # A user-defined main() is renamed so it cannot clash with the runner's.
        # Results go to a dup of the original stdout; fd 1 is then pointed at
        # stderr so the solution's own prints (cout/printf) never corrupt them.
        return f'''#include <bits/stdc++.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
using namespace std;
#define main leetcode_user_main
#include "solution.cpp"
#undef main
int main() {{
    FILE* leetcode_out = fdopen(dup(1), "w");
    dup2(2, 1);
    {instance}
    string line;
    while (getline(cin, line)) {{
        auto args = nlohmann::json::parse(line);
        {declarations}
        auto result = {target}({names});
        cout.flush();
        fprintf(leetcode_out, "%s\\n", nlohmann::json({output}).dump().c_str());
        fflush(leetcode_out);
    }}
}}
'''
    if language == "go":
        output = "string([]byte{result})" if interface["return_type"] == "byte" else "result"
        declarations = "\n".join(
            f"var arg{i} {p['type']}; if err := json.Unmarshal(args[{i}], &arg{i}); err != nil {{ panic(err) }}"
            for i, p in enumerate(params)
        )
        # The encoder keeps the real stdout; os.Stdout is then swapped to stderr
        # so fmt.Print* calls in the solution do not corrupt the results.
        return f'''package main
import ("encoding/json"; "os"; "io"; "reflect")
// A nil Go slice represents an empty LeetCode array, not Python None.
func leetcodeJSONValue(value interface{{}}) interface{{}} {{
    v := reflect.ValueOf(value)
    if v.Kind() == reflect.Slice {{
        out := make([]interface{{}}, v.Len())
        for i := range out {{ out[i] = leetcodeJSONValue(v.Index(i).Interface()) }}
        return out
    }}
    return value
}}
func main() {{
    decoder := json.NewDecoder(os.Stdin)
    encoder := json.NewEncoder(os.Stdout)
    os.Stdout = os.Stderr
    for {{
        var args []json.RawMessage
        if err := decoder.Decode(&args); err == io.EOF {{ return }} else if err != nil {{ panic(err) }}
        {declarations}
        result := {call}({names})
        if err := encoder.Encode(leetcodeJSONValue({output})); err != nil {{ panic(err) }}
    }}
}}
'''
    if language == "java":
        declarations = "\n".join(
            f"{p['type']} arg{i} = gson.fromJson(args.get({i}), new com.google.gson.reflect.TypeToken<{ {'int':'Integer','long':'Long','double':'Double','boolean':'Boolean','char':'Character','float':'Float'}.get(p['type'],p['type'])}>() {{}}.getType());"
            for i, p in enumerate(params)
        )
        target = f"solution.{call}" if container else call
        instance = f"{container} solution = new {container}();" if container else ""
        return f'''import java.util.*;
import com.google.gson.*;
class Runner {{
    public static void main(String[] unused) {{
        // Keep the real stdout for results; the solution's prints go to stderr.
        java.io.PrintStream leetcodeOut = System.out;
        System.setOut(System.err);
        Gson gson = new Gson();
        {instance}
        Scanner input = new Scanner(System.in);
        while (input.hasNextLine()) {{
            JsonArray args = JsonParser.parseString(input.nextLine()).getAsJsonArray();
            {declarations}
            var result = {target}({names});
            leetcodeOut.println(gson.toJson(result));
        }}
    }}
}}
'''
    if language == "rust":
        declarations = "\n".join(
            (
                f"let arg{i}: {p['type']} = "
                f"serde_json::from_value(args[{i}].clone())"
                f".expect(\"failed to deserialize argument {i}\");"
            )
            for i, p in enumerate(params)
        )

        target = (
            f"{container}::{call}"
            if container
            else call
        )

        # Runner imports live inside main() so they cannot collide with the
        # solution's own `use` lines (include! shares the module scope). The
        # judge drops `struct Solution;` / rewrites the call when the solution
        # defines its own struct or free function. Results go to a dup of the
        # original stdout; fd 1 is pointed at stderr for the solution's prints.
        return f'''#[allow(unused_imports)]
use std::collections::*;

extern "C" {{
    fn dup(fd: i32) -> i32;
    fn dup2(src: i32, dst: i32) -> i32;
}}

struct Solution;

include!("solution.rs");

fn main() {{
    use std::io::{{BufRead, Write}};
    use std::os::unix::io::FromRawFd;

    let mut leetcode_out = unsafe {{
        let fd = dup(1);
        dup2(2, 1);
        std::fs::File::from_raw_fd(fd)
    }};
    let stdin = std::io::stdin();

    for line in stdin.lock().lines() {{
        let line = line.expect("failed to read stdin");

        if line.trim().is_empty() {{
            continue;
        }}

        let args: Vec<serde_json::Value> =
            serde_json::from_str(&line)
                .expect("failed to parse arguments");

        {declarations}

        let result = {target}({names});

        writeln!(
            leetcode_out,
            "{{}}",
            serde_json::to_string(&result)
                .expect("failed to serialize result")
        )
        .expect("failed to write result");
        leetcode_out.flush().expect("failed to flush result");
    }}
}}
'''
    if language == "javascript":
        # The solution may define the entry point as a declaration (function,
        # var, let, const), export it via module.exports/exports, or wrap it in
        # a Solution class. Its console writes to stderr, not the result stream,
        # and LeetCode's JS libraries are available as globals.
        return f'''const fs = require("fs");
const readline = require("readline");
const vm = require("vm");
const {{ Console }} = require("console");

const solutionCode = fs.readFileSync("solution.js", "utf8");

// Libraries LeetCode provides as globals (PriorityQueue, Queue, Deque, _).
function leetcodeRequire(name) {{
    try {{
        return require(name);
    }} catch (error) {{
        return require("/usr/local/lib/node_modules/" + name);
    }}
}}
function leetcodeLibrary(name) {{
    try {{
        return leetcodeRequire(name);
    }} catch (error) {{
        return {{}};
    }}
}}

const solutionModule = {{ exports: {{}} }};
const context = {{
    ...leetcodeLibrary("@datastructures-js/priority-queue"),
    ...leetcodeLibrary("@datastructures-js/queue"),
    ...leetcodeLibrary("@datastructures-js/deque"),
    _: leetcodeLibrary("lodash"),
    module: solutionModule,
    exports: solutionModule.exports,
    require: Object.assign(leetcodeRequire, {{ main: undefined }}),
    console: new Console({{ stdout: process.stderr, stderr: process.stderr }}),
}};
vm.createContext(context);
vm.runInContext(solutionCode, context, {{ filename: "solution.js" }});

function findTarget() {{
    // Top-level let/const are not context properties, but they are visible
    // to later scripts run in the same context.
    const declared = vm.runInContext(
        'typeof {call} === "function" ? {call} : undefined',
        context,
    );
    if (declared) return declared;
    const exported = solutionModule.exports;
    if (typeof exported === "function") return exported;
    if (exported && typeof exported["{call}"] === "function") return exported["{call}"];
    const solutionClass = vm.runInContext(
        'typeof Solution === "function" ? Solution : undefined',
        context,
    );
    if (solutionClass && typeof solutionClass.prototype["{call}"] === "function") {{
        const instance = new solutionClass();
        return instance["{call}"].bind(instance);
    }}
    return undefined;
}}

const target = findTarget();

if (typeof target !== "function") {{
    throw new Error("Expected function {call} was not defined");
}}

const rl = readline.createInterface({{
    input: process.stdin,
    crlfDelay: Infinity,
}});

rl.on("line", (line) => {{
    if (!line.trim()) {{
        return;
    }}

    const args = JSON.parse(line);
    const result = target(...args);

    process.stdout.write(
        JSON.stringify(result) + "\\n"
    );
}});
'''

    if language == "typescript":
        return f'''declare function require(name: string): any;
declare const process: any;

const readline = require("readline");
const target: any = {call};

const rl = readline.createInterface({{
    input: process.stdin,
    crlfDelay: Infinity,
}});

rl.on("line", (line: string) => {{
    if (!line.trim()) {{
        return;
    }}

    const args = JSON.parse(line);
    const result = target(...args);

    process.stdout.write(
        JSON.stringify(result) + "\\n"
    );
}});
'''
    if language == "php":
        # Accept the entry point as a Solution method or a free function. Any
        # output the solution prints is captured and forwarded to stderr.
        if container:
            target = (
                f"if (class_exists(\"{container}\") || !function_exists(\"{call}\")) {{\n"
                f"        $solver = new {container}();\n"
                f"        $result = $solver->{call}(...$args);\n"
                f"    }} else {{\n"
                f"        $result = {call}(...$args);\n"
                f"    }}"
            )
        else:
            target = f"$result = {call}(...$args);"

        return f'''<?php

ob_start();
require_once "solution.php";
fwrite(STDERR, ob_get_clean());

while (($line = fgets(STDIN)) !== false) {{
    $line = trim($line);

    if ($line === "") {{
        continue;
    }}

    $args = json_decode($line, true);

    ob_start();
    {target}
    fwrite(STDERR, ob_get_clean());

    fwrite(STDOUT, json_encode($result) . PHP_EOL);
}}
'''
    if language == "ruby":
        # Accept the entry point as a top-level method, a Solution instance
        # method, or a Solution class method. $stdout is pointed at stderr so
        # the solution's puts/print never corrupt the results on STDOUT.
        if container:
            target = (
                f"solver = {container}.new\n"
                f"  result = solver.{call}(*args)"
            )
        else:
            target = f"result = leetcode_target.call(*args)"

        return f'''require "json"

STDOUT.sync = true
$stdout = STDERR

require_relative "solution"

leetcode_target =
  if respond_to?(:{call}, true)
    method(:{call})
  elsif defined?(Solution) && Solution.method_defined?(:{call})
    Solution.new.method(:{call})
  elsif defined?(Solution) && Solution.respond_to?(:{call})
    Solution.method(:{call})
  end

while (line = STDIN.gets)
  line = line.strip

  next if line.empty?

  args = JSON.parse(line)

  {target}

  STDOUT.puts(JSON.generate(result))
end
'''
    raise ValueError(language)
