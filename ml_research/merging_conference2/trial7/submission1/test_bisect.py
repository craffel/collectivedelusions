import subprocess

def test_compile(content):
    with open("test_sub.tex", "w") as f:
        f.write(content)
    res = subprocess.run(["tectonic", "test_sub.tex"], capture_output=True, text=True)
    return res.returncode == 0, res.stderr

with open("submission.tex", "r") as f:
    orig_lines = f.readlines()

# Let's write a script that helps us bisect the lines of the document
# We want to keep the preamble and the document end, but vary the content in \begin{document} ... \end{document}
preamble = []
end_doc = ["\\end{document}\n"]
doc_body = []

in_body = False
for line in orig_lines:
    if "\\begin{document}" in line:
        preamble.append(line)
        in_body = True
        continue
    if "\\end{document}" in line:
        in_body = False
        continue
    if in_body:
        doc_body.append(line)
    else:
        preamble.append(line)

print(f"Preamble: {len(preamble)} lines, Body: {len(doc_body)} lines")

# Let's test if compiling with empty body works
test_body = []
full_test = "".join(preamble) + "".join(test_body) + "".join(end_doc)
ok, err = test_compile(full_test)
print("Empty body compilation:", ok)
if not ok:
    print("Empty body error:", err)

# Let's test compiling with first half of the body
test_body = doc_body[:len(doc_body)//2]
full_test = "".join(preamble) + "".join(test_body) + "".join(end_doc)
ok, err = test_compile(full_test)
print("First half compilation:", ok)

# Let's test compiling with second half of the body
test_body = doc_body[len(doc_body)//2:]
full_test = "".join(preamble) + "".join(test_body) + "".join(end_doc)
ok, err = test_compile(full_test)
print("Second half compilation:", ok)
