è necessario che ci sia una funzione separata che si occupa solo della correzione, presi in input i dati da correggere. fix_issue() magari può diventare la funzione che le accorpa insieme (così non bisogna modificare i tests?).fix_issue() chiamerebbe le funzioni separate tenendo presente il checkpoint. **FORSE NON è NECESSARIO**.


1. Il checkpoint diventa un attributo della classe madre ProvenanceFixer (checkpoint=CheckpointManager("prov_fix_ckp.json")).

2. Quando un detect_issue() viene ultimato correttamente, si salva un checkpoint che dice che la fase di detect di un determinato fixer è stata completata. Se invece ci sono interruzioni durante la fase di detect, non viene scritto niente. Questo può essere fatto all'interno di fix_issue!

3. checkpointed_batch rimane così com'è (?)

4. fix_issue() tiene in considerazione il checkpoint prima di lanciare eventualmente detect_issue(): se la fase di detect è ultimata, la salta (e avvisa l'utente con un warning o con un info nel log); se invece il processo si è interrotto nella fase di correzione, riprende dal batch successivo a quello in cui si è interrotto il processo.

5. fix_process() semplicemente chiama i fix_issue() di ciascun fixer in sequenza, poi se si giunge senza errori o interruzioni alla fine di fix_process, il checkpoint viene silenziosamente eliminato.





3277432
37702