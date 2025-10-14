
# rsync -av Book/ Course_Website/
# cd Course_Website
# cd Book

rm -rf docs
cp -a _book docs
date=$(date '+%Y-%m-%d')
git add *
git commit -a -m "Dispatch book $date"
git push > /dev/null 2>&1
