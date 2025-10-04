è necessario che ci sia una funzione separata che si occupa solo della correzione, presi in input i dati da correggere. fix_issue() magari può diventare la funzione che le accorpa insieme (così non bisogna modificare i tests?).fix_issue() chiamerebbe le funzioni separate tenendo presente il checkpoint. **FORSE NON è NECESSARIO**.


1. Il checkpoint diventa un attributo della classe madre ProvenanceFixer (checkpoint=CheckpointManager("prov_fix_ckp.json")).

2. Quando un detect_issue() viene ultimato correttamente, si salva un checkpoint che dice che la fase di detect di un determinato fixer è stata completata. Se invece ci sono interruzioni durante la fase di detect, non viene scritto niente. Questo può essere fatto all'interno di fix_issue!

3. checkpointed_batch rimane così com'è (?)

4. fix_issue() tiene in considerazione il checkpoint prima di lanciare eventualmente detect_issue(): se la fase di detect è ultimata, la salta (e avvisa l'utente con un warning o con un info nel log); se invece il processo si è interrotto nella fase di correzione, riprende dal batch successivo a quello in cui si è interrotto il processo.

5. fix_process() semplicemente chiama i fix_issue() di ciascun fixer in sequenza, poi se si giunge senza errori o interruzioni alla fine di fix_process, il checkpoint viene silenziosamente eliminato.





3277432
37702



Il DB Virtuoso deve essere configurato in virtuoso.ini, nella sezione **SPARQL** con un valore di MaxResultRows superiore o uguale al LIMIT delle query SELECT, altrimenti vengono persi dei risultati. 



Da documentazione di virutoso.ini (https://docs.openlinksw.com/virtuoso/ch-server/#ini_sparql):


[SPARQL]

* ResultSetMaxRows = number .  This setting is used to limit the number of the rows in the result. The effective limit will be the lowest of this setting, SPARQL query 'LIMIT' clause value (if present), and SPARQL Endpoint request URI &maxrows parameter value (if present).

* MaxQueryCostEstimationTime = seconds .  This setting is used to limit the estimate time cost of the query to certain number of seconds, the default is no limit.

* MaxQueryExecutionTime = seconds .  This setting is used to set the transaction execution timeout to certain limit in number of seconds, the default is no limit.


---------------------------------------

**Il parametro "ResultSetMaxRows" nella sezione [Parameters] di virtuoso.ini secondo me non fa niente, perché è un parametro per la sezione [SPARQL], non per la sezione [Parameters] (non è documentato per la sezione [Parameters]). In virtuoso_utilities viene configurato a 100'000, ma solo per la sezione [Parameters]: questo valore viene poi "sovrascritto" (in realtà, credo, semplicemente ignorato) in favore del valore di default della sezione [SPARQL], che è 10'000!**. Vedi queste righe di codice: 

* https://github.com/opencitations/virtuoso_utilities/blob/c64de093894cb7f8aca7b6f61989ab91040c2154/virtuoso_utilities/launch_virtuoso.py#L27
* https://github.com/opencitations/virtuoso_utilities/blob/c64de093894cb7f8aca7b6f61989ab91040c2154/virtuoso_utilities/launch_virtuoso.py#L598
* https://github.com/opencitations/virtuoso_utilities/blob/c64de093894cb7f8aca7b6f61989ab91040c2154/virtuoso_utilities/launch_virtuoso.py#L617

-----------------------------------------

MaxSortedTopRows = 10000.  The TOP select statement clause caches in memory the rows pertinent to the result. The number of rows allowed to be cached within memory is limited by this parameter.

Simple example using OFFSET and LIMIT:

Virtuoso uses a zero index in the OFFSET. Thus in the example below, will be taken position at record 9000 in the result set, and will get the next 1000 rows starting from 9001 record. Note that the MaxSortedTopRows in parameters Virtuoso ini section needs to be increased (default is 10000).

    select ?name
    ORDER BY ?name
    OFFSET 9000
    LIMIT 1000






# TODO
* prova il detect con LIMIT 10'000, vedi se ci sono tutte le righe e  MaxSortedTopRows si comporta bene. 
* poi prova il detect con un LIMIT più alto ma alzando solo ResultSetMaxRows nella sezione [SPARQL], lasciando  così com'è.
    * il mio dubbio è che MaxSortedTopRows sia già gestito (rispetto al ritornare tutti i risultati, non rispetto all'efficienza) dallo scrollable cursor (praticamente, la subquery indicata qui: https://vos.openlinksw.com/owiki/wiki/VOS/VirtTipsAndTricksHowToHandleBandwidthLimitExceed), e credo invece che il problema stia nel fatto che i risultati della query siano limitati da ResultSetMaxRows. Anche se il limit della query è iù alto, Virutoso ritorna il limite configurato in INI senza dire che mancano delle righe nei risultati. In altre parole, per ogni singla query della paginazione, se il LIMIT della query è più alto del valore di [SPARQL]ResultSetMaxRows in INI, Virtuoso ritorna comunque solo _N_ risultati (dove _N_ == [SPARQL]ResultSetMaxRows in INI), senza dire niente. Ad esempio, se LIMIT==100'000 e SPARQL]ResultSetMaxRows==10'000, e mi aspetto 1'000'000 risultati, Virtuoso farà comunque 1000000/100000=**10** query, ma per ogni query ritornerà solo i primi 10000 risultati, arrivando in totale a 100000 risultati e lasciandone fuori 900'000.
* se non funziona (anche senza dare errori), cioè se non tutte le righe del count sono presenti, considera di usare la keyset pagination. Vedi anche quanto tempo ci mette



**NB**: Non mi è chiaro se Virtuoso, anche con le subqueries indicate [qui](https://vos.openlinksw.com/owiki/wiki/VOS/VirtTipsAndTricksHowToHandleBandwidthLimitExceed), deve materializzare _TUTTI_ i risultati prima di ordinarli. Se consì fosse, potrebbe comunque non bastare la memoria. In tal caso, bisognerebbe davvero riscrivere il codice usato una keyset pagination (sempre che quella funzioni).


Vedi qui per keyset pagination di esempio:
https://chatgpt.com/c/68de9973-e948-8331-b9af-c2064107134b

