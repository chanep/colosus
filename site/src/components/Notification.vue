<template>
    <p class="error">
        {{gameStatus.error}}
    </p>
</template>

<script>
    import store from "../store"

    export default {
        data () {
            return {
                gameStatus: store.gameStatus,
                }
            }
    }
</script>

<style scoped>
    .error {
        font-size: medium;
        color: red;
    }
</style>