if [ "$1" == "t" ] ; then
$(stack path --local-install-root)/bin/lojban-test #+RTS -p
else
echo "Starting Chat Bot"
$(stack path --local-install-root)/bin/lojbanChatBot #+RTS -xc -RTS
fi
