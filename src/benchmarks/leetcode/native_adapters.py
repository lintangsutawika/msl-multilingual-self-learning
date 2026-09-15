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
    raise ValueError(language)
