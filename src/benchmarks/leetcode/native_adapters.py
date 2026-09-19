"""Generate JSON-lines workers using the native Doocs types and entrypoints."""
from __future__ import annotations


def render_worker(problem, language):
    interface = problem["interfaces"][language]
    params = interface["parameters"]
    names = ", ".join(f"arg{i}" for i in range(len(params)))
    call = interface["callable"]
    container = interface.get("container")
    if language == "cpp":
        declarations = "\n".join(
            f"auto arg{i} = args.at({i}).get<{p['type'].replace('&', '').replace('const ', '').strip()}>();"
            for i, p in enumerate(params)
        )
        target = f"solution.{call}" if container else call
        instance = f"{container} solution;" if container else ""
        output = "string(1, result)" if interface["return_type"] == "char" else "result"
        return f'''#include <bits/stdc++.h>
#include <nlohmann/json.hpp>
using namespace std;
#include "solution.cpp"
int main() {{
    {instance}
    string line;
    while (getline(cin, line)) {{
        auto args = nlohmann::json::parse(line);
        {declarations}
        auto result = {target}({names});
        cout << nlohmann::json({output}).dump() << endl;
    }}
}}
'''
    if language == "go":
        output = "string([]byte{result})" if interface["return_type"] == "byte" else "result"
        declarations = "\n".join(
            f"var arg{i} {p['type']}; if err := json.Unmarshal(args[{i}], &arg{i}); err != nil {{ panic(err) }}"
            for i, p in enumerate(params)
        )
        return f'''package main
import ("encoding/json"; "os"; "io"; "reflect")
// A nil Go slice represents an empty LeetCode array, not Python None.
func jsonValue(value interface{{}}) interface{{}} {{
    v := reflect.ValueOf(value)
    if v.Kind() == reflect.Slice {{
        out := make([]interface{{}}, v.Len())
        for i := range out {{ out[i] = jsonValue(v.Index(i).Interface()) }}
        return out
    }}
    return value
}}
func main() {{
    decoder := json.NewDecoder(os.Stdin)
    encoder := json.NewEncoder(os.Stdout)
    for {{
        var args []json.RawMessage
        if err := decoder.Decode(&args); err == io.EOF {{ return }} else if err != nil {{ panic(err) }}
        {declarations}
        result := {call}({names})
        if err := encoder.Encode(jsonValue({output})); err != nil {{ panic(err) }}
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
        Gson gson = new Gson();
        {instance}
        Scanner input = new Scanner(System.in);
        while (input.hasNextLine()) {{
            JsonArray args = JsonParser.parseString(input.nextLine()).getAsJsonArray();
            {declarations}
            var result = {target}({names});
            System.out.println(gson.toJson(result));
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

        return f'''use std::io::{{self, BufRead}};

struct Solution;

include!("solution.rs");

fn main() {{
    let stdin = io::stdin();

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

        println!(
            "{{}}",
            serde_json::to_string(&result)
                .expect("failed to serialize result")
        );
    }}
}}
'''
    if language == "javascript":
        return f'''const fs = require("fs");
const readline = require("readline");
const vm = require("vm");

const solutionCode = fs.readFileSync("solution.js", "utf8");

const context = {{}};
vm.createContext(context);
vm.runInContext(solutionCode, context);

const target = context["{call}"];

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
        if container:
            target = (
                f"$solver = new {container}();\n"
                f"$result = $solver->{call}(...$args);"
            )
        else:
            target = f"$result = {call}(...$args);"

        return f'''<?php

require_once "solution.php";

while (($line = fgets(STDIN)) !== false) {{
    $line = trim($line);

    if ($line === "") {{
        continue;
    }}

    $args = json_decode($line, true);

    {target}

    fwrite(STDOUT, json_encode($result) . PHP_EOL);
}}
'''
    if language == "ruby":
        if container:
            target = (
                f"solver = {container}.new\n"
                f"result = solver.{call}(*args)"
            )
        else:
            target = f"result = {call}(*args)"

        return f'''require "json"
require_relative "solution"

STDOUT.sync = true

while (line = STDIN.gets)
  line = line.strip

  next if line.empty?

  args = JSON.parse(line)

  {target}

  STDOUT.puts(JSON.generate(result))
end
'''
    raise ValueError(language)
