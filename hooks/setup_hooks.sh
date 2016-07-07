# Don't forget to run this before testing :)
# (Copies all hooks to .git/hooks (where they will actually be called))
echo Setting up hooks
cp * ../.git/hooks
cp .get_branch ../.git/hooks
# Remove ourselves from the hooks directory 
rm "../.git/hooks/$0"
echo Hook setup completed
