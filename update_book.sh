
# rsync -av Book/ Course_Website/
# cd Course_Website
# cd Book

cd _book 
# Remove all HTML comments (<!-- ... -->) from HTML files in this directory

# Find all .html files recursively
find . -type f -name "*.html" | while read -r file; do
    echo "Cleaning comments in: $file"
    # Use Perl for robust multiline removal
    perl -0777 -pe 's/<!--.*?-->//sg' "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"
done
cd ../

echo "Succesfully removed all HTML comments."

rm -rf docs
mv _book docs
date=$(date '+%Y-%m-%d')
git add *
git commit -a -m "Dispatch book $date"
git push
